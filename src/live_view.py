"""Live measurement view: the application/entry logic for the vision system.

This module is deliberately free of any camera-specific (Aravis) configuration.
It drives any *frame source* that implements ``get_frame()`` and ``close()``
(for example :class:`src.camera_interface.AravisCamera` or
:class:`src.camera_interface.FileFrameSource`), runs the measurement pipeline,
renders the overlay, handles keyboard controls, and can log accuracy trials.

Camera hardware setup (exposure, gain, white balance, device selection) lives in
the launcher scripts, not here, so this module can be explained on its own.
"""

from datetime import datetime
from pathlib import Path
import time
from typing import Optional, Protocol, Sequence, Tuple, cast, runtime_checkable

import cv2
import numpy as np

from config.system_config import default_config
from src.accuracy_log import AccuracyLog, match_ground_truth
from src.pipeline import MeasurementPipeline, SceneResult


@runtime_checkable
class FrameSource(Protocol):
    """Minimal interface the live view needs from a frame source."""

    def get_frame(self) -> np.ndarray:
        ...

    def close(self) -> None:
        ...


def _to_bgr(frame: np.ndarray) -> np.ndarray:
    """Return a BGR copy of a grayscale or color frame."""
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return frame.copy()


def _status_frame(frame: np.ndarray) -> np.ndarray:
    """Frame with a 'Searching for nail...' status hint."""
    display = _to_bgr(frame)
    cv2.putText(display, "Searching for nail...", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)
    return display


def _render_frame(
    frame: np.ndarray,
    pipeline: Optional[MeasurementPipeline],
    ground_truth_mm: Optional[float] = None,
    condition: Optional[str] = None,
    reference_index: int = 0,
    ground_truth_presets: Optional[Sequence[float]] = None,
) -> Tuple[np.ndarray, Optional[SceneResult]]:
    """Render one display frame; returns the image and the scene (if measured)."""
    if pipeline is None:
        return _to_bgr(frame), None

    try:
        scene = pipeline.process_all(
            frame,
            ground_truth_mm=ground_truth_mm,
            condition=condition,
            reference_index=reference_index,
            ground_truth_presets=ground_truth_presets,
        )
        if scene.annotated_image is not None:
            return scene.annotated_image, scene
        return _to_bgr(frame), scene
    except Exception:
        return _status_frame(frame), None


def _draw_status_line(display_frame: np.ndarray, text: str) -> None:
    cv2.putText(display_frame, text, (20, display_frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)


def _draw_fps(display_frame: np.ndarray, fps: float, frame: np.ndarray) -> None:
    cv2.putText(
        display_frame,
        f"FPS: {fps:.1f} | Frame: {frame.shape[1]}x{frame.shape[0]}",
        (20, display_frame.shape[0] - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def run_live_view(
    source: FrameSource,
    *,
    enable_measurement: bool = False,
    display_width: int = 1280,
    save_dir: Optional[str] = None,
    window_name: str = "Live Measurement View",
    ground_truth_presets: Optional[Sequence[float]] = None,
    condition_presets: Optional[Sequence[str]] = None,
    initial_ground_truth: Optional[float] = None,
    initial_condition: Optional[str] = None,
    auto_match: bool = False,
    log_csv: Optional[str] = None,
) -> None:
    """Display a live frame source with an optional measurement overlay.

    Keyboard controls (the window must be focused):
        'q' / ESC : exit and release the source
        's'       : save the current full-resolution frame
        'm'       : toggle the measurement overlay
        'e'       : run auto exposure + gain once (if the source supports it)
        'w'       : run auto white balance once (if the source supports it)
        'g'       : cycle the ground-truth preset length
        'c'       : cycle the illumination-condition label
        TAB / 'n' : select the next detected object as the reference
        digits/.  : type a custom ground-truth length, ENTER to apply
        'r'       : record the current measurement for the accuracy table
        'a'       : toggle auto exposure on/off
        '[' / ']' : decrease / increase exposure manually
        '-' / '=' : decrease / increase gain manually
        't'       : toggle auto-match ground truth (each nail vs nearest preset)
    """
    presets = list(ground_truth_presets if ground_truth_presets is not None else default_config.demo.ground_truth_presets_mm)
    conditions = list(condition_presets if condition_presets is not None else default_config.demo.condition_presets)

    gt_index = 0
    ground_truth: Optional[float] = initial_ground_truth if initial_ground_truth is not None else (presets[0] if presets else None)
    cond_index = 0
    condition: Optional[str] = initial_condition if initial_condition is not None else (conditions[0] if conditions else None)
    typed_gt = ""
    reference_index = 0
    auto_exposure_on = True
    exposure_text = ""
    auto_match_mode = auto_match

    log = AccuracyLog(log_csv) if log_csv else None

    print("=" * 65)
    print("LIVE MEASUREMENT VIEW")
    print("=" * 65)

    info: dict = {}
    get_info = getattr(source, "get_info", None)
    if callable(get_info):
        try:
            info = cast(dict, get_info() or {})
        except Exception:
            info = {}
    if info:
        print(f"Source:          {info.get('model', 'Camera')} ({info.get('device_id', 'N/A')})")
        print(f"Resolution:      {info.get('width', '?')} x {info.get('height', '?')}")

    print(f"Measurement:     {'ACTIVE' if enable_measurement else 'OFF (press m to toggle)'}")
    print(f"Ground truth:    {ground_truth if ground_truth is not None else 'none'} mm (press g to cycle, type a value + Enter)")
    print(f"Condition:       {condition if condition else 'none'} (press c to cycle)")
    if log is not None:
        print(f"Accuracy log:    {log.csv_path}")
    print("\nControls (click the window first so it receives key presses):")
    print("  'q' / ESC : Exit")
    print("  's'       : Save current frame")
    print("  'm'       : Toggle measurement overlay")
    print("  'e'       : Run auto exposure + gain once")
    print("  'w'       : Run auto white balance once")
    print("  'g'       : Cycle ground-truth preset")
    print("  'c'       : Cycle condition label")
    print("  TAB / 'n' : Select next object as reference")
    print("  0-9 . Enter : Type a custom ground-truth length")
    print("  'r'       : Record current measurement (accuracy table)")
    print("  'a'       : Toggle auto exposure on/off")
    print("  '[' / ']' : Decrease / increase exposure")
    print("  '-' / '=' : Decrease / increase gain")
    print("  't'       : Toggle auto-match ground truth (per nail)")
    print("=" * 65)

    save_path = Path(save_dir) if save_dir else None
    if save_path is not None:
        save_path.mkdir(parents=True, exist_ok=True)

    pipeline = MeasurementPipeline() if enable_measurement else None
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    # On macOS the window must be focused to receive key presses; bring it to front.
    cv2.moveWindow(window_name, 40, 40)
    try:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
    except Exception:
        pass

    frame_count = 0
    t_start = time.time()
    fps = 0.0
    last_scene: Optional[SceneResult] = None

    try:
        while True:
            try:
                frame = source.get_frame()
            except TimeoutError:
                continue
            except Exception as err:
                print(f"Acquisition error: {err}")
                break

            # Optional host-side auto exposure (only if the source provides it).
            update_auto = getattr(source, "update_auto_exposure", None)
            if callable(update_auto):
                update_auto(frame)

            frame_count += 1
            if frame_count % 15 == 0:
                dt = time.time() - t_start
                if dt > 0:
                    fps = 15.0 / dt
                t_start = time.time()
                get_info_fn = getattr(source, "get_info", None)
                if callable(get_info_fn):
                    try:
                        sinfo = cast(dict, get_info_fn() or {})
                        exp = sinfo.get("exposure_time_us")
                        gain = sinfo.get("gain_db")
                        exp_str = f"{exp:.0f}us" if exp is not None else "n/a"
                        gain_str = f"{gain:.1f}dB" if gain is not None else "n/a"
                        exposure_text = f"Exp {exp_str} Gain {gain_str} auto={'on' if auto_exposure_on else 'off'}"
                    except Exception:
                        exposure_text = ""

            display_frame, last_scene = _render_frame(
                frame,
                pipeline,
                ground_truth,
                condition,
                reference_index,
                ground_truth_presets=presets if auto_match_mode else None,
            )
            _draw_fps(display_frame, fps, frame)
            gt_text = f"GT: {ground_truth:.1f} mm" if ground_truth is not None else "GT: none"
            count = len(last_scene.measurements) if last_scene is not None else 0
            ref_index = min(reference_index, count - 1) if count else 0
            ref_text = f"Ref: #{ref_index + 1}/{count}" if count else "Ref: -"
            mode_text = "auto-match" if auto_match_mode else "reference"
            _draw_status_line(
                display_frame,
                f"GT mode: {mode_text} | {gt_text} | Cond: {condition or 'none'} | {ref_text} | "
                f"{exposure_text} | typing: {typed_gt or '-'}",
            )

            h, w = display_frame.shape[:2]
            scale = display_width / float(w)
            screen_img = cv2.resize(
                display_frame,
                (display_width, int(round(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
            cv2.imshow(window_name, screen_img)

            key = cv2.waitKey(10) & 0xFF
            if key in (27, ord("q")):
                break
            elif key == ord("s"):
                if save_path is not None:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    snap_file = save_path / f"nail_{timestamp}.png"
                    cv2.imwrite(str(snap_file), frame)
                    print(f"Saved snapshot: {snap_file}")
                else:
                    print("No save directory configured.")
            elif key == ord("m"):
                pipeline = MeasurementPipeline() if pipeline is None else None
                print(f"Measurement overlay: {'ENABLED' if pipeline is not None else 'DISABLED'}")
            elif key == ord("e"):
                run_once = getattr(source, "run_auto_once", None)
                if callable(run_once) and run_once():
                    print("Auto exposure/gain: running once...")
                else:
                    print("Auto exposure/gain: not available on this source.")
            elif key == ord("w"):
                set_wb = getattr(source, "set_white_balance_auto", None)
                if callable(set_wb) and set_wb("once"):
                    print("White balance: running once...")
                else:
                    print("White balance: not available on this source.")
            elif key == ord("a"):
                set_auto = getattr(source, "set_auto_exposure_enabled", None)
                if callable(set_auto):
                    auto_exposure_on = not auto_exposure_on
                    set_auto(auto_exposure_on)
                    print(f"Auto exposure: {'ON' if auto_exposure_on else 'OFF'}")
                else:
                    print("Auto exposure toggle not available on this source.")
            elif key == ord("["):
                nudge = getattr(source, "nudge_exposure", None)
                if callable(nudge) and nudge(0.8):
                    auto_exposure_on = False
                    print("Exposure: decreased")
                else:
                    print("Exposure control not available on this source.")
            elif key == ord("]"):
                nudge = getattr(source, "nudge_exposure", None)
                if callable(nudge) and nudge(1.25):
                    auto_exposure_on = False
                    print("Exposure: increased")
                else:
                    print("Exposure control not available on this source.")
            elif key == ord("-"):
                nudge = getattr(source, "nudge_gain", None)
                if callable(nudge) and nudge(-1.0):
                    print("Gain: decreased")
                else:
                    print("Gain control not available on this source.")
            elif key == ord("="):
                nudge = getattr(source, "nudge_gain", None)
                if callable(nudge) and nudge(1.0):
                    print("Gain: increased")
                else:
                    print("Gain control not available on this source.")
            elif key == ord("g"):
                if presets:
                    gt_index = (gt_index + 1) % len(presets)
                    ground_truth = presets[gt_index]
                    typed_gt = ""
                    print(f"Ground truth: {ground_truth:.1f} mm")
            elif key == ord("c"):
                if conditions:
                    cond_index = (cond_index + 1) % len(conditions)
                    condition = conditions[cond_index]
                    print(f"Condition: {condition}")
            elif key in (9, ord("n")):  # TAB or 'n': next reference object
                reference_index += 1
                print(f"Reference: #{reference_index + 1}")
            elif key == ord("t"):
                auto_match_mode = not auto_match_mode
                print(f"Ground-truth mode: {'auto-match (per nail)' if auto_match_mode else 'single reference'}")
            elif key in (8, 127):  # backspace
                typed_gt = typed_gt[:-1]
            elif key in (10, 13):  # enter: apply typed value
                if typed_gt:
                    try:
                        ground_truth = float(typed_gt)
                        print(f"Ground truth (custom): {ground_truth:.2f} mm")
                    except ValueError:
                        print(f"Invalid ground truth: {typed_gt!r}")
                    typed_gt = ""
            elif 48 <= key <= 57 or key == ord("."):
                typed_gt += chr(key)
            elif key == ord("r"):
                if pipeline is None or last_scene is None or not last_scene.measurements:
                    print("Nothing to record (no measurement).")
                elif log is None:
                    print("No accuracy log configured.")
                elif auto_match_mode:
                    recorded = 0
                    for idx, meas in enumerate(last_scene.measurements):
                        matched_gt = match_ground_truth(meas.length_mm, presets)
                        if matched_gt is None:
                            continue
                        log.add(
                            condition=condition,
                            object_id=f"#{idx + 1}",
                            measured_px=meas.length_px,
                            measured_mm=meas.length_mm,
                            ground_truth_mm=matched_gt,
                        )
                        recorded += 1
                    print(f"Recorded {recorded} auto-matched nail(s) -> {log.csv_path}")
                else:
                    ref = min(reference_index, len(last_scene.measurements) - 1)
                    reference = last_scene.measurements[ref]
                    log.add(
                        condition=condition,
                        object_id=f"#{ref + 1}",
                        measured_px=reference.length_px,
                        measured_mm=reference.length_mm,
                        ground_truth_mm=ground_truth,
                    )
                    print(
                        f"Recorded #{ref + 1}: {reference.length_mm:.2f} mm "
                        f"(GT {ground_truth if ground_truth is not None else '-'}) -> {log.csv_path}"
                    )

    finally:
        source.close()
        cv2.destroyAllWindows()
        if log is not None and log.rows:
            print("\n" + "=" * 70)
            print("ACCURACY SUMMARY (live demo)")
            print("=" * 70)
            print(log.format_table())
            print(f"Saved to: {log.csv_path}")
        print("Source released. Exited cleanly.")
