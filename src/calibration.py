"""Pixel-to-millimeter calibration module.

Two complementary calibration paths are provided:

1. **Projection-derived** (default): the scale is computed from the camera
   projection model in ``docs/projection-note.md`` using focal length, working
   distance and sensor geometry. See :func:`projection_scale`.
2. **Empirical**: the scale is measured from a known reference object using
   :func:`calculate_scale`.

Both produce a single ``mm_per_pixel`` factor used by the measurement pipeline.
"""

from typing import Optional

from config.system_config import default_config


def default_mm_per_pixel() -> float:
    """Resolve the active calibration scale (explicit override or projection)."""
    value = default_config.calibration.mm_per_pixel
    if value is None:
        value = default_config.camera.mm_per_pixel
    return float(value)


_DEFAULT_MM_PER_PIXEL: float = default_mm_per_pixel()


def projection_scale(camera=default_config.camera) -> float:
    """Projection-derived mm_per_pixel for the configured camera geometry."""
    return camera.mm_per_pixel


def pixels_to_mm(pixels: float, mm_per_pixel: float = _DEFAULT_MM_PER_PIXEL) -> float:
    """Convert image pixel measurement into physical millimeters."""
    return float(pixels * mm_per_pixel)


def mm_to_pixels(mm: float, mm_per_pixel: float = _DEFAULT_MM_PER_PIXEL) -> float:
    """Convert physical millimeters into image pixels."""
    if mm_per_pixel <= 0:
        raise ValueError(f"Invalid mm_per_pixel: {mm_per_pixel}")
    return float(mm / mm_per_pixel)


def calculate_scale(known_length_mm: float, measured_pixels: float) -> float:
    """Calculate empirical mm_per_pixel scale factor from a known reference measurement."""
    if known_length_mm <= 0 or measured_pixels <= 0:
        raise ValueError("Lengths and pixel measurements must be positive.")
    return float(known_length_mm / measured_pixels)


class PixelCalibration:
    """Simple container for pixel-to-millimeter calibration."""

    def __init__(self, mm_per_pixel: Optional[float] = None) -> None:
        if mm_per_pixel is None:
            mm_per_pixel = _DEFAULT_MM_PER_PIXEL
        if mm_per_pixel <= 0:
            raise ValueError(f"Invalid scale factor: {mm_per_pixel}")
        self.mm_per_pixel = float(mm_per_pixel)

    def to_mm(self, pixels: float) -> float:
        return pixels_to_mm(pixels, self.mm_per_pixel)

    def to_pixels(self, mm: float) -> float:
        return mm_to_pixels(mm, self.mm_per_pixel)
