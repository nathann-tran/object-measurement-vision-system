"""Central system configuration for the Object Measurement Vision System."""

from dataclasses import dataclass, field
from typing import Optional

from src.projection import mm_per_pixel_from_camera


@dataclass(frozen=True)
class CameraConfig:
    """Hardware specifications for IDS U3-33F0XCP-C-HQ camera and optical setup.

    Sensor geometry is taken from the Sony IMX568 datasheet (optical area
    6.773 x 5.650 mm, 2.74 um pixels) and matches the configured resolution.
    """

    image_width_px: int = 2472
    image_height_px: int = 2062
    focal_length_mm: float = 8.0
    working_distance_mm: float = 397.0
    sensor_width_mm: float = 6.773
    sensor_height_mm: float = 5.650
    pixel_size_um: float = 2.74

    @property
    def mm_per_pixel(self) -> float:
        """Projection-derived scale (mm per pixel) for this optical setup."""
        return mm_per_pixel_from_camera(
            focal_length_mm=self.focal_length_mm,
            working_distance_mm=self.working_distance_mm,
            sensor_width_mm=self.sensor_width_mm,
            sensor_height_mm=self.sensor_height_mm,
            image_width_px=self.image_width_px,
            image_height_px=self.image_height_px,
        )


@dataclass(frozen=True)
class NailGroundTruth:
    """Ground-truth lengths for reference nails (caliper measurements in mm)."""

    small_mm: float = 24.0
    medium_mm: float = 30.0
    large_mm: float = 55.0


@dataclass
class CalibrationConfig:
    """Pixel-to-millimeter calibration setting.

    ``mm_per_pixel`` is derived from the camera projection model (see
    ``docs/projection-note.md``). When left as ``None`` it is filled from the
    active :class:`CameraConfig` in :meth:`SystemConfig.__post_init__`. Set it
    explicitly to override the projection scale with an empirical value obtained
    from a known reference object (see ``src.calibration.calculate_scale``).
    """

    mm_per_pixel: Optional[float] = None


@dataclass
class SegmentationConfig:
    """Parameters for segmentation and morphological cleanup.

    ``threshold_method`` selects global Otsu (``"otsu"``) or local adaptive
    thresholding (``"adaptive"``). Adaptive is the default because it tolerates
    strong / non-uniform illumination (e.g. a ring light or a bright highlight)
    where a single global threshold breaks down.
    """

    gaussian_blur_kernel: int = 5
    threshold_method: str = "adaptive"  # "adaptive" or "otsu"
    adaptive_block_size: int = 101  # must be odd and larger than the object width
    adaptive_c: float = 10.0
    min_object_area_px: float = 500.0
    max_object_area_fraction: float = 0.90  # Reject near whole-frame fills
    max_object_extent_fraction: float = 0.90  # Reject contours spanning the full frame
    max_objects: int = 20  # Cap the number of detected objects per frame
    invert_threshold: bool = True  # True if nail is darker than illuminated background
    morphology_kernel_size: int = 3


@dataclass
class SystemConfig:
    """Unified system configuration."""

    camera: CameraConfig = field(default_factory=CameraConfig)
    ground_truth: NailGroundTruth = field(default_factory=NailGroundTruth)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)

    def __post_init__(self) -> None:
        if self.calibration.mm_per_pixel is None:
            self.calibration.mm_per_pixel = self.camera.mm_per_pixel


default_config = SystemConfig()
