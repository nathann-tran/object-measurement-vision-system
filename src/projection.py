"""Camera projection utilities for the object-measurement system.

Implements the pinhole projection result documented in ``docs/projection-note.md``:
the physical millimetres-per-pixel scale for a fixed camera/object distance.

These helpers are intentionally pure (no configuration or hardware imports) so
both ``config.system_config`` and ``src.calibration`` can reuse them without
creating an import cycle.
"""


def mm_per_pixel(
    real_distance_mm: float,
    focal_length_mm: float,
    sensor_size_mm: float,
    sensor_size_px: float,
) -> float:
    """Physical millimetres per image pixel along one sensor axis.

    Derived from the projection chain for a 1 mm object at ``real_distance_mm``:

        mm_per_pixel = real_distance * sensor_size_mm / (f * sensor_size_px)
    """
    if focal_length_mm <= 0:
        raise ValueError(f"Focal length must be positive, got {focal_length_mm}.")
    if sensor_size_px <= 0:
        raise ValueError(f"Sensor resolution must be positive, got {sensor_size_px}.")
    return real_distance_mm * sensor_size_mm / (focal_length_mm * sensor_size_px)


def mm_per_pixel_from_camera(
    focal_length_mm: float,
    working_distance_mm: float,
    sensor_width_mm: float,
    sensor_height_mm: float,
    image_width_px: int,
    image_height_px: int,
) -> float:
    """Average the horizontal and vertical projection scales for a camera setup.

    The two axes should agree when the sensor dimensions and image resolution
    describe the same pixel pitch; averaging removes small datasheet rounding.
    """
    scale_x = mm_per_pixel(working_distance_mm, focal_length_mm, sensor_width_mm, image_width_px)
    scale_y = mm_per_pixel(working_distance_mm, focal_length_mm, sensor_height_mm, image_height_px)
    return (scale_x + scale_y) / 2.0
