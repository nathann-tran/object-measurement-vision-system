"""Pixel-to-millimeter calibration.

Resolves the active ``mm_per_pixel`` scale — either an explicit override or the
projection-derived value from the camera geometry — and converts pixel
measurements to millimetres.
"""

from typing import Optional

from config.system_config import default_config


def default_mm_per_pixel() -> float:
    """Resolve the active calibration scale (explicit override or projection)."""
    value = default_config.calibration.mm_per_pixel
    if value is None:
        value = default_config.camera.mm_per_pixel
    return float(value)


def pixels_to_mm(pixels: float, mm_per_pixel: Optional[float] = None) -> float:
    """Convert an image pixel measurement into physical millimeters."""
    if mm_per_pixel is None:
        mm_per_pixel = default_mm_per_pixel()
    return float(pixels * mm_per_pixel)
