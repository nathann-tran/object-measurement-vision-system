"""Camera projection utilities derived from the projection notebook.

Implements the pinhole projection chain documented in ``docs/projection-note.md``:

    real-world size
        -> perspective projection (size on the sensor, mm)
        -> physical sensor geometry (mm)
        -> image coordinates (px)

These helpers are intentionally pure (no configuration or hardware imports) so
both ``config.system_config`` and ``src.calibration`` can reuse them without
creating an import cycle.
"""

from typing import Tuple


def pinhole_project(
    x: float,
    y: float,
    z: float,
    focal_length_mm: float,
) -> Tuple[float, float]:
    """Project a 3D camera-space point onto the sensor plane (pinhole model).

    ``x' = f * X / Z`` and ``y' = f * Y / Z``.
    """
    if z <= 0:
        raise ValueError(f"Point must be in front of the camera (Z > 0), got Z={z}.")
    return focal_length_mm * x / z, focal_length_mm * y / z


def projected_size_mm(
    real_size_mm: float,
    real_distance_mm: float,
    focal_length_mm: float,
) -> float:
    """Projected physical size on the sensor for an object at a given distance.

    ``projected_size_mm = f * real_size / real_distance``.
    """
    if real_distance_mm <= 0:
        raise ValueError(f"Distance must be positive, got {real_distance_mm}.")
    return focal_length_mm * real_size_mm / real_distance_mm


def sensor_mm_to_px(
    size_mm: float,
    sensor_size_mm: float,
    sensor_size_px: float,
) -> float:
    """Map a physical sensor dimension in millimetres to image pixels.

    Uses the proportional mapping ``fraction of physical sensor == fraction of
    image resolution``. The axis of ``size_mm`` must match ``sensor_size_mm``.
    """
    if sensor_size_mm <= 0:
        raise ValueError(f"Sensor dimension must be positive, got {sensor_size_mm}.")
    return size_mm / sensor_size_mm * sensor_size_px


def projected_size_px(
    real_size_mm: float,
    real_distance_mm: float,
    focal_length_mm: float,
    sensor_size_mm: float,
    sensor_size_px: float,
) -> float:
    """Compose perspective projection and sensor mm -> px conversion."""
    return sensor_mm_to_px(
        projected_size_mm(real_size_mm, real_distance_mm, focal_length_mm),
        sensor_size_mm,
        sensor_size_px,
    )


def mm_per_pixel(
    real_distance_mm: float,
    focal_length_mm: float,
    sensor_size_mm: float,
    sensor_size_px: float,
) -> float:
    """Physical millimetres per image pixel along one sensor axis.

    Inverts :func:`projected_size_px` for a 1 mm object at ``real_distance_mm``:

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
