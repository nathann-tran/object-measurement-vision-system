"""Visualization module for measurement overlays using standard OpenCV drawing.

Renders:
1. Object boundary contour (green).
2. Rotated bounding rectangle (blue/cyan).
3. Longitudinal caliper measurement line & endpoints (yellow & red).
4. Text HUD displaying measured length(s) in mm and pixels, plus ground truth/error.
"""

from typing import List, Optional, Sequence
import cv2
import numpy as np

from src.measurement import NailMeasurement

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _prepare_overlay(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image.copy()


def _draw_nail(
    overlay: np.ndarray,
    contour: np.ndarray,
    measurement: NailMeasurement,
    length_mm: Optional[float] = None,
) -> None:
    """Draw one nail's contour, bounding box, caliper line, and length label."""
    cv2.drawContours(overlay, [contour], -1, (0, 255, 0), 2)

    box_int = np.int32(np.round(measurement.box_points))
    cv2.polylines(overlay, [box_int], isClosed=True, color=(255, 180, 0), thickness=1, lineType=cv2.LINE_AA)  # type: ignore

    ep1 = (int(round(measurement.endpoint1[0])), int(round(measurement.endpoint1[1])))
    ep2 = (int(round(measurement.endpoint2[0])), int(round(measurement.endpoint2[1])))
    cv2.line(overlay, ep1, ep2, (0, 255, 255), 2, lineType=cv2.LINE_AA)
    cv2.circle(overlay, ep1, 5, (0, 0, 255), -1)
    cv2.circle(overlay, ep2, 5, (0, 0, 255), -1)

    if length_mm is not None:
        # Label just above the topmost corner of the rotated bounding box.
        top = measurement.box_points[np.argmin(measurement.box_points[:, 1])]
        _draw_label(overlay, f"{length_mm:.1f} mm", int(top[0]), int(top[1]) - 8)


def _draw_label(overlay: np.ndarray, text: str, x: int, y: int) -> None:
    """Draw a small high-contrast label with a filled background box."""
    scale = 0.6
    (tw, th), baseline = cv2.getTextSize(text, _FONT, scale, 1)
    x = int(max(0, min(x - tw // 2, overlay.shape[1] - tw - 6)))
    y = int(max(th + 6, y))
    cv2.rectangle(overlay, (x - 3, y - th - 5), (x + tw + 3, y + baseline - 1), (20, 20, 20), -1)
    cv2.putText(overlay, text, (x, y), _FONT, scale, (0, 255, 255), 1, cv2.LINE_AA)


def _draw_hud(overlay: np.ndarray, lines: Sequence[tuple[str, tuple[int, int, int], float]]) -> None:
    """Draw a translucent HUD banner with (text, color, scale) rows."""
    height = 30 + 30 * len(lines)
    hud_bg = overlay.copy()
    cv2.rectangle(hud_bg, (15, 15), (560, 15 + height), (25, 25, 25), -1)
    cv2.addWeighted(hud_bg, 0.7, overlay, 0.3, 0, overlay)
    cv2.rectangle(overlay, (15, 15), (560, 15 + height), (100, 100, 100), 1)

    y = 45
    for text, color, scale in lines:
        cv2.putText(overlay, text, (25, y), _FONT, scale, color, 2, cv2.LINE_AA)
        y += 30


def draw_measurement(
    image: np.ndarray,
    contour: np.ndarray,
    measurement: NailMeasurement,
    length_mm: float,
    ground_truth_mm: Optional[float] = None,
    condition: Optional[str] = None,
) -> np.ndarray:
    """Render a single-object measurement annotation and HUD."""
    overlay = _prepare_overlay(image)
    _draw_nail(overlay, contour, measurement, length_mm=length_mm)

    lines = [
        (f"Length: {length_mm:.2f} mm ({measurement.length_px:.1f} px)", (0, 255, 255), 0.75),
    ]
    if ground_truth_mm is not None:
        abs_err = abs(length_mm - ground_truth_mm)
        rel_err = (abs_err / ground_truth_mm) * 100.0
        err_color = (0, 255, 0) if rel_err < 2.0 else (0, 140, 255)
        lines.append(
            (
                f"Ground Truth: {ground_truth_mm:.1f} mm | Error: {abs_err:.2f} mm ({rel_err:.1f}%)",
                err_color,
                0.6,
            )
        )
    if condition:
        lines.append((f"Condition: {condition}", (200, 200, 200), 0.55))

    _draw_hud(overlay, lines)
    return overlay


def draw_measurements(
    image: np.ndarray,
    contours: List[np.ndarray],
    measurements: List[NailMeasurement],
    length_mms: List[float],
    ground_truth_mm: Optional[float] = None,
    condition: Optional[str] = None,
) -> np.ndarray:
    """Render every detected object and a summary HUD."""
    overlay = _prepare_overlay(image)

    for contour, measurement, length_mm in zip(contours, measurements, length_mms):
        _draw_nail(overlay, contour, measurement, length_mm=length_mm)

    lines = [(f"Detected: {len(measurements)} object(s)", (0, 255, 255), 0.75)]
    if ground_truth_mm is not None and length_mms:
        largest = max(length_mms)
        abs_err = abs(largest - ground_truth_mm)
        rel_err = (abs_err / ground_truth_mm) * 100.0
        lines.append(
            (
                f"Largest vs GT {ground_truth_mm:.1f} mm | Err {abs_err:.2f} mm ({rel_err:.1f}%)",
                (0, 255, 0) if rel_err < 2.0 else (0, 140, 255),
                0.55,
            )
        )
    if condition:
        lines.append((f"Condition: {condition}", (200, 200, 200), 0.55))

    _draw_hud(overlay, lines)
    return overlay


def create_diagnostic_view(
    image: np.ndarray,
    mask: np.ndarray,
    annotated: np.ndarray,
    target_width: int = 1600,
) -> np.ndarray:
    """Create a side-by-side diagnostic image: [Original | Mask | Annotated]."""
    if image.ndim == 2:
        img_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        img_bgr = image.copy()

    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    combined = np.hstack([img_bgr, mask_bgr, annotated])
    if target_width > 0 and combined.shape[1] != target_width:
        h, w = combined.shape[:2]
        scale = target_width / float(w)
        combined = cv2.resize(combined, (target_width, int(round(h * scale))), interpolation=cv2.INTER_AREA)

    return combined
