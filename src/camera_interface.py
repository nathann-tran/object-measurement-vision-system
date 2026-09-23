"""Camera acquisition interface using Aravis and OpenCV.

Handles physical IDS camera connection, ROI configuration, buffer streaming,
and conversion to 2D NumPy arrays. Also provides a file reader for offline testing.

Exposure notes:
- This IDS U3-33FxXCP-C exposes ``ExposureTime`` and ``Gain`` but no
  ``ExposureAuto``/``GainAuto`` GenICam features, so auto exposure is performed
  on the host by measuring frame brightness and adjusting ``ExposureTime``
  (and ``Gain`` once exposure saturates). See :meth:`AravisCamera.update_auto_exposure`.
"""

from pathlib import Path
from typing import Any, Optional, Union
import math

import cv2
import numpy as np

from config.system_config import default_config

_AUTO_MODES = {"off": "Off", "once": "Once", "continuous": "Continuous"}


class AravisCamera:
    """Industrial camera acquisition interface via Aravis 0.8."""

    def __init__(
        self,
        camera_id: Optional[str] = None,
        target_width: int = default_config.camera.image_width_px,
        target_height: int = default_config.camera.image_height_px,
        buffer_count: int = 3,
        auto_exposure: Optional[str] = None,
        auto_gain: Optional[str] = None,
        exposure_time_us: Optional[float] = None,
        gain_db: Optional[float] = None,
        exposure_target: float = 110.0,
        exposure_tolerance: float = 12.0,
        white_balance: Optional[str] = None,
        white_balance_ratios: Optional[tuple[float, float, float]] = None,
    ) -> None:
        self.target_width = target_width
        self.target_height = target_height
        self.buffer_count = buffer_count
        self.device_id: Optional[str] = None
        # Aravis/GObject objects are dynamic (no Python stubs), so they are Any.
        self._aravis: Any = None
        self.camera: Any = None
        self.stream: Any = None

        # Host-side auto exposure/gain state.
        self.exposure_target = float(exposure_target)
        self.exposure_tolerance = float(exposure_tolerance)
        self.auto_exposure_interval = 3  # adjust every N frames
        self._auto_exposure_mode: Optional[str] = None
        self._auto_gain_mode: Optional[str] = None
        self._exposure_bounds: Optional[tuple[float, float]] = None
        self._gain_bounds: Optional[tuple[float, float]] = None
        self._exposure_time: Optional[float] = None
        self._gain: Optional[float] = None
        self._ae_frame_count = 0

        self._init_aravis()
        self._open_camera(camera_id)

        self._exposure_bounds = self._read_float_bounds("ExposureTime")
        self._gain_bounds = self._read_float_bounds("Gain")
        self._exposure_time = self._get_float_feature("ExposureTime")
        self._gain = self._get_float_feature("Gain")

        self.configure_exposure(
            auto_exposure=auto_exposure,
            auto_gain=auto_gain,
            exposure_time_us=exposure_time_us,
            gain_db=gain_db,
        )
        self.configure_white_balance(mode=white_balance, ratios=white_balance_ratios)
        self._setup_stream()

    def _init_aravis(self) -> None:
        try:
            import gi
            gi.require_version("Aravis", "0.8")
            from gi.repository import Aravis  # type: ignore
            self._aravis = Aravis
        except (ImportError, ValueError) as err:
            raise RuntimeError(
                f"Aravis 0.8 not found. Please verify PyGObject and Aravis installation: {err}"
            ) from err

    def _open_camera(self, camera_id: Optional[str]) -> None:
        Aravis = self._aravis
        Aravis.update_device_list()
        if Aravis.get_n_devices() == 0:
            raise RuntimeError("No Aravis camera detected on USB/GigE interfaces.")

        device_id = camera_id or Aravis.get_device_id(0)
        self.camera = Aravis.Camera.new(device_id)
        if self.camera is None:
            raise RuntimeError(f"Failed to open camera: {device_id}")
        self.device_id = device_id

        # Set ROI to official 2472 x 2062 sensor resolution
        try:
            self.camera.set_region(0, 0, self.target_width, self.target_height)
        except Exception:
            pass

    def _setup_stream(self) -> None:
        Aravis = self._aravis
        payload = self.camera.get_payload()
        self.stream = self.camera.create_stream(None, None)
        if self.stream is None:
            raise RuntimeError("Failed to create Aravis stream.")

        for _ in range(self.buffer_count):
            buf = Aravis.Buffer.new_allocate(payload)
            self.stream.push_buffer(buf)

        self.camera.start_acquisition()

    # ------------------------------------------------------------------
    # GenICam feature helpers (Aravis binding: get_*/set_* by feature name)
    # ------------------------------------------------------------------
    def _feature_available(self, name: str) -> bool:
        checker = getattr(self.camera, "is_feature_available", None)
        if checker is None:
            return True
        try:
            return bool(checker(name))
        except Exception:
            return False

    def _set_string_feature(self, name: str, value: str) -> bool:
        try:
            self.camera.set_string(name, value)
            return True
        except Exception:
            return False

    def _get_string_feature(self, name: str) -> Optional[str]:
        try:
            return self.camera.get_string(name)
        except Exception:
            return None

    def _set_float_feature(self, name: str, value: float) -> bool:
        try:
            self.camera.set_float(name, float(value))
            return True
        except Exception:
            return False

    def _get_float_feature(self, name: str) -> Optional[float]:
        try:
            return float(self.camera.get_float(name))
        except Exception:
            return None

    def _read_float_bounds(self, name: str) -> Optional[tuple[float, float]]:
        try:
            bounds = self.camera.get_float_bounds(name)
            return float(bounds[0]), float(bounds[1])
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Camera-side auto exposure/gain (only if the camera supports it)
    # ------------------------------------------------------------------
    def _set_auto_mode(self, feature: str, mode: str, label: str) -> bool:
        mode_key = mode.lower()
        if mode_key not in _AUTO_MODES:
            raise ValueError(
                f"Unsupported {label} mode: {mode!r}. Choose from {sorted(_AUTO_MODES)}."
            )
        if not self._feature_available(feature):
            return False
        return self._set_string_feature(feature, _AUTO_MODES[mode_key])

    def set_auto_exposure(self, mode: str = "continuous") -> bool:
        """Set camera-side ``ExposureAuto`` (``"off"``, ``"once"`` or ``"continuous"``)."""
        return self._set_auto_mode("ExposureAuto", mode, "auto exposure")

    def set_auto_gain(self, mode: str = "continuous") -> bool:
        """Set camera-side ``GainAuto`` (``"off"``, ``"once"`` or ``"continuous"``)."""
        return self._set_auto_mode("GainAuto", mode, "auto gain")

    # ------------------------------------------------------------------
    # Manual exposure/gain
    # ------------------------------------------------------------------
    def get_exposure_time(self) -> Optional[float]:
        value = self._get_float_feature("ExposureTime")
        if value is not None:
            self._exposure_time = value
        return value

    def set_exposure_time(self, microseconds: float) -> bool:
        """Set a fixed exposure time in microseconds."""
        if self._set_float_feature("ExposureTime", microseconds):
            self._exposure_time = float(microseconds)
            return True
        return False

    def get_gain(self) -> Optional[float]:
        value = self._get_float_feature("Gain")
        if value is not None:
            self._gain = value
        return value

    def set_gain(self, db: float) -> bool:
        """Set a fixed gain in dB."""
        if self._set_float_feature("Gain", db):
            self._gain = float(db)
            return True
        return False

    # ------------------------------------------------------------------
    # Host-side auto exposure/gain
    # ------------------------------------------------------------------
    def enable_software_auto_exposure(
        self,
        mode: str = "continuous",
        target: Optional[float] = None,
        tolerance: Optional[float] = None,
    ) -> bool:
        """Enable host-side auto exposure (adjusts ``ExposureTime`` per frame)."""
        mode_key = mode.lower()
        if mode_key not in _AUTO_MODES:
            raise ValueError(
                f"Unsupported auto exposure mode: {mode!r}. Choose from {sorted(_AUTO_MODES)}."
            )
        if target is not None:
            self.exposure_target = float(target)
        if tolerance is not None:
            self.exposure_tolerance = float(tolerance)
        self._auto_exposure_mode = None if mode_key == "off" else mode_key
        self._ae_frame_count = 0
        return self._exposure_bounds is not None

    def enable_software_auto_gain(self, mode: str = "continuous") -> bool:
        """Enable host-side auto gain (used once exposure saturates at its max)."""
        mode_key = mode.lower()
        if mode_key not in _AUTO_MODES:
            raise ValueError(
                f"Unsupported auto gain mode: {mode!r}. Choose from {sorted(_AUTO_MODES)}."
            )
        self._auto_gain_mode = None if mode_key == "off" else mode_key
        return self._gain_bounds is not None

    def update_auto_exposure(self, frame: np.ndarray) -> None:
        """Adjust exposure/gain toward the target brightness using a frame."""
        if self._auto_exposure_mode not in ("continuous", "once"):
            return

        self._ae_frame_count += 1
        if self._ae_frame_count % self.auto_exposure_interval != 0:
            return

        mean = float(np.mean(frame))
        if abs(mean - self.exposure_target) <= self.exposure_tolerance:
            if self._auto_exposure_mode == "once":
                self._auto_exposure_mode = "off"
            if self._auto_gain_mode == "once":
                self._auto_gain_mode = "off"
            return

        ratio = self.exposure_target / max(mean, 1.0)
        ratio = min(max(ratio, 0.5), 2.0)  # limit each step to avoid oscillation
        self._apply_exposure_ratio(ratio)

    def _apply_exposure_ratio(self, ratio: float) -> None:
        exposure = self._exposure_time if self._exposure_time is not None else self.get_exposure_time()
        if exposure is None or self._exposure_bounds is None:
            return
        lo, hi = self._exposure_bounds
        desired = exposure * ratio
        target = min(max(desired, lo), hi)
        if self._set_float_feature("ExposureTime", target):
            self._exposure_time = target
        # If exposure hit a limit, cover the remaining factor with gain.
        if self._auto_gain_mode in ("continuous", "once"):
            if desired > hi:
                self._apply_gain_ratio(desired / hi)
            elif desired < lo:
                self._apply_gain_ratio(desired / lo)

    def _apply_gain_ratio(self, ratio: float) -> None:
        gain = self._gain if self._gain is not None else self.get_gain()
        if gain is None or self._gain_bounds is None:
            return
        lo, hi = self._gain_bounds
        # Gain is in dB: multiplying the linear signal by `ratio` adds 20*log10(ratio).
        desired = gain + 20.0 * math.log10(max(ratio, 1e-3))
        target = min(max(desired, lo), hi)
        if self._set_float_feature("Gain", target):
            self._gain = target

    def configure_exposure(
        self,
        auto_exposure: Optional[str] = None,
        auto_gain: Optional[str] = None,
        exposure_time_us: Optional[float] = None,
        gain_db: Optional[float] = None,
    ) -> bool:
        """Apply exposure/gain settings.

        Manual values are applied first. Auto modes prefer the camera's own
        ``ExposureAuto``/``GainAuto`` when available, otherwise host-side auto
        exposure/gain is enabled.
        """
        applied = False
        if exposure_time_us is not None:
            applied = self.set_exposure_time(exposure_time_us) or applied
        if gain_db is not None:
            applied = self.set_gain(gain_db) or applied

        if auto_exposure is not None:
            if self._feature_available("ExposureAuto") and self.set_auto_exposure(auto_exposure):
                pass
            else:
                applied = self.enable_software_auto_exposure(auto_exposure) or applied

        if auto_gain is not None:
            if self._feature_available("GainAuto") and self.set_auto_gain(auto_gain):
                pass
            else:
                applied = self.enable_software_auto_gain(auto_gain) or applied

        return applied

    def run_auto_once(self) -> bool:
        """Run one auto exposure/gain pass for the current scene."""
        if self._feature_available("ExposureAuto"):
            applied = self.set_auto_exposure("once")
        else:
            applied = self.enable_software_auto_exposure("once")

        if self._feature_available("GainAuto"):
            applied = self.set_auto_gain("once") or applied
        elif self._auto_gain_mode is not None:
            applied = self.enable_software_auto_gain("once") or applied
        return applied

    # ------------------------------------------------------------------
    # White balance (only if the camera supports it)
    # ------------------------------------------------------------------
    def set_white_balance_auto(self, mode: str = "continuous") -> bool:
        """Set camera-side ``BalanceWhiteAuto`` (``"off"``, ``"once"`` or ``"continuous"``)."""
        return self._set_auto_mode("BalanceWhiteAuto", mode, "white balance")

    def set_white_balance_ratios(self, ratios: tuple[float, float, float]) -> bool:
        """Set manual red/green/blue balance ratios (disables auto white balance)."""
        if not self._feature_available("BalanceRatio"):
            return False
        self._set_string_feature("BalanceWhiteAuto", "Off")
        applied = False
        for selector, value in zip(("Red", "Green", "Blue"), ratios):
            if not self._set_string_feature("BalanceRatioSelector", selector):
                continue
            if self._set_float_feature("BalanceRatio", value):
                applied = True
        return applied

    def configure_white_balance(
        self,
        mode: Optional[str] = None,
        ratios: Optional[tuple[float, float, float]] = None,
    ) -> bool:
        """Apply white balance settings.

        Manual ``ratios`` take precedence; otherwise the auto ``mode`` is used.
        A ``None`` mode leaves the camera's factory default untouched.
        """
        if ratios is not None:
            return self.set_white_balance_ratios(ratios)
        if mode is not None:
            return self.set_white_balance_auto(mode)
        return False

    def get_white_balance_mode(self) -> Optional[str]:
        """Return the current ``BalanceWhiteAuto`` mode, or None if unavailable."""
        return self._get_string_feature("BalanceWhiteAuto")

    # ------------------------------------------------------------------
    # Acquisition
    # ------------------------------------------------------------------
    def get_frame(self) -> np.ndarray:
        """Acquire a single frame and convert to a grayscale NumPy array (2062, 2472)."""
        Aravis = self._aravis
        buf = self.stream.timeout_pop_buffer(2_000_000)
        if buf is None:
            raise TimeoutError("Camera frame acquisition timed out.")

        try:
            if buf.get_status() != Aravis.BufferStatus.SUCCESS:
                raise RuntimeError(f"Buffer acquisition error: {buf.get_status()}")

            try:
                buf_w = buf.get_image_width()
                buf_h = buf.get_image_height()
                pad = buf.get_image_padding()
                x_pad = getattr(pad, "x_padding", 0) if pad is not None else 0
            except Exception:
                buf_w, buf_h, x_pad = 0, 0, 0

            img_w = buf_w if buf_w > 0 else self.target_width
            img_h = buf_h if buf_h > 0 else self.target_height

            try:
                raw_data = buf.get_image_data()
            except Exception:
                raw_data = buf.get_data()

            raw_array = np.frombuffer(raw_data, dtype=np.uint8)
            row_stride = img_w + x_pad
            required_bytes = img_h * row_stride

            if raw_array.size >= required_bytes and x_pad > 0:
                raw_matrix = raw_array[:required_bytes].reshape((img_h, row_stride))
                image = raw_matrix[:, :img_w].copy()
            else:
                image = raw_array[:img_h * img_w].reshape((img_h, img_w))

            # Crop if buffer exceeds expected dimensions
            if image.shape[0] > self.target_height or image.shape[1] > self.target_width:
                image = image[:self.target_height, :self.target_width]

            # Convert Bayer to grayscale
            return cv2.cvtColor(image, cv2.COLOR_BayerRG2GRAY)

        finally:
            self.stream.push_buffer(buf)

    def close(self) -> None:
        """Cleanly stop stream and release camera resources."""
        if self.camera is not None:
            try:
                self.camera.stop_acquisition()
            except Exception:
                pass
        self.stream = None
        self.camera = None

    def get_info(self) -> dict:
        if self.camera is None:
            return {}
        # Querying the camera's GenICam `DeviceID` feature can fail on some
        # models while streaming, so reuse the ID we opened the camera with.
        try:
            model = self.camera.get_model_name()
        except Exception:
            model = "Camera"
        return {
            "model": model,
            "device_id": self.device_id,
            "width": self.target_width,
            "height": self.target_height,
            "auto_exposure": self._get_string_feature("ExposureAuto") or self._auto_exposure_mode,
            "auto_gain": self._get_string_feature("GainAuto") or self._auto_gain_mode,
            "exposure_time_us": self._get_float_feature("ExposureTime"),
            "gain_db": self._get_float_feature("Gain"),
            "white_balance": self.get_white_balance_mode(),
        }


class FileFrameSource:
    """Offline frame reader loading static images from disk via OpenCV."""

    def __init__(self, image_path: Union[Path, str]) -> None:
        self.image_path = Path(image_path)
        if not self.image_path.exists():
            raise FileNotFoundError(f"File not found: {self.image_path}")

    def get_frame(self) -> np.ndarray:
        image = cv2.imread(str(self.image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to read image from {self.image_path}")
        return image

    def close(self) -> None:
        pass
