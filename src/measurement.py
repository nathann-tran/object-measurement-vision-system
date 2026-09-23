"""Geometric measurement of segmented objects using standard OpenCV geometry.

Uses cv2.minAreaRect to determine the rotated bounding rectangle, longitudinal
pixel length, width, orientation, and caliper measurement endpoints.
"""

from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class NailMeasurement:
    """Measurement results and geometry for a detected nail."""

    length_px: float
    width_px: float
    center: tuple[float, float]
    angle_deg: float
    box_points: np.ndarray  # Shape (4, 2) corner coordinates
    endpoint1: tuple[float, float]  # First caliper endpoint
    endpoint2: tuple[float, float]  # Second caliper endpoint


def measure_nail(contour: np.ndarray) -> NailMeasurement:
    """Measure the longitudinal extent of the nail using cv2.minAreaRect.

    Args:
        contour: Object boundary contour from cv2.findContours.

    Returns:
        NailMeasurement: Measured pixel dimensions, orientation, and caliper endpoints.
    """
    rect = cv2.minAreaRect(contour)
    (cx, cy), (w, h), angle = rect

    length_px = float(max(w, h))
    width_px = float(min(w, h))

    # Obtain the 4 corner vertices of the rotated rectangle
    box = cv2.boxPoints(rect)  # Array of 4 points in clockwise order

    # Identify the two short edges to find longitudinal caliper endpoints (midpoints of short edges)
    # Edge lengths between consecutive corners
    edge0 = float(np.linalg.norm(box[1] - box[0]))
    edge1 = float(np.linalg.norm(box[2] - box[1]))

    if edge0 < edge1:
        # box[0]-box[1] and box[2]-box[3] are the short edges
        ep1 = tuple(((box[0] + box[1]) / 2.0).astype(float))
        ep2 = tuple(((box[2] + box[3]) / 2.0).astype(float))
    else:
        # box[1]-box[2] and box[3]-box[0] are the short edges
        ep1 = tuple(((box[1] + box[2]) / 2.0).astype(float))
        ep2 = tuple(((box[3] + box[0]) / 2.0).astype(float))

    return NailMeasurement(
        length_px=length_px,
        width_px=width_px,
        center=(float(cx), float(cy)),
        angle_deg=float(angle),
        box_points=box,
        endpoint1=(float(ep1[0]), float(ep1[1])),
        endpoint2=(float(ep2[0]), float(ep2[1])),
    )
