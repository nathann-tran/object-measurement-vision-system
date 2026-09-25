"""End-to-end measurement pipeline orchestration module.

Orchestrates the conventional visual measurement flow:
Frame -> Segmentation -> Measurement (cv2.minAreaRect) -> Calibration (px to mm) -> Visualization.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Sequence
import numpy as np

from config.system_config import default_config
from src.segmentation import segment_nail, segment_nails
from src.measurement import NailMeasurement, measure_nail
from src.calibration import default_mm_per_pixel, pixels_to_mm
from src.accuracy_log import match_ground_truth
from src.visualization import draw_measurement, draw_measurements


@dataclass
class MeasurementResult:
    """Consolidated outcome for a single measured object."""

    length_px: float
    length_mm: float
    width_mm: float
    contour: np.ndarray
    mask: np.ndarray
    measurement: NailMeasurement
    annotated_image: Optional[np.ndarray] = None
    ground_truth_mm: Optional[float] = None
    absolute_error_mm: Optional[float] = None
    relative_error_percent: Optional[float] = None


@dataclass
class SceneResult:
    """Outcome for a full frame that may contain several objects."""

    measurements: List[MeasurementResult] = field(default_factory=list)
    mask: Optional[np.ndarray] = None
    annotated_image: Optional[np.ndarray] = None

    @property
    def count(self) -> int:
        return len(self.measurements)


class MeasurementPipeline:
    """Orchestrates image processing, measurement, calibration, and rendering."""

    def __init__(
        self,
        mm_per_pixel: float = default_mm_per_pixel(),
        min_area: float = default_config.segmentation.min_object_area_px,
        invert_threshold: bool = default_config.segmentation.invert_threshold,
        threshold_method: str = default_config.segmentation.threshold_method,
        adaptive_block_size: int = default_config.segmentation.adaptive_block_size,
        adaptive_c: float = default_config.segmentation.adaptive_c,
    ) -> None:
        self.mm_per_pixel = mm_per_pixel
        self.min_area = min_area
        self.invert_threshold = invert_threshold
        self.threshold_method = threshold_method
        self.adaptive_block_size = adaptive_block_size
        self.adaptive_c = adaptive_c

    def   _segment(self, image: np.ndarray):
        return segment_nails(
            image,
            method=self.threshold_method,
            block_size=self.adaptive_block_size,
            c=self.adaptive_c,
            min_area=self.min_area,
            invert=self.invert_threshold,
        )

    @staticmethod
    def _errors(length_mm: float, ground_truth_mm: Optional[float]):
        if ground_truth_mm is not None and ground_truth_mm > 0:
            abs_err = abs(length_mm - ground_truth_mm)
            rel_err = (abs_err / ground_truth_mm) * 100.0
            return abs_err, rel_err
        return None, None

    def _measure(self, contour, mask, ground_truth_mm):
        meas = measure_nail(contour)
        length_mm = pixels_to_mm(meas.length_px, self.mm_per_pixel)
        width_mm = pixels_to_mm(meas.width_px, self.mm_per_pixel)
        abs_err, rel_err = self._errors(length_mm, ground_truth_mm)
        return MeasurementResult(
            length_px=meas.length_px,
            length_mm=length_mm,
            width_mm=width_mm,
            contour=contour,
            mask=mask,
            measurement=meas,
            ground_truth_mm=ground_truth_mm,
            absolute_error_mm=abs_err,
            relative_error_percent=rel_err,
        )

    def process(
        self,
        image: np.ndarray,
        ground_truth_mm: Optional[float] = None,
        condition: Optional[str] = None,
    ) -> MeasurementResult:
        """Measure the single largest object in the frame (backward-compatible)."""
        contour, mask = segment_nail(
            image,
            method=self.threshold_method,
            block_size=self.adaptive_block_size,
            c=self.adaptive_c,
            min_area=self.min_area,
            invert=self.invert_threshold,
        )
        result = self._measure(contour, mask, ground_truth_mm)
        result.annotated_image = draw_measurement(
            image=image,
            contour=contour,
            measurement=result.measurement,
            length_mm=result.length_mm,
            ground_truth_mm=ground_truth_mm,
            condition=condition,
        )
        return result

    def process_all(
        self,
        image: np.ndarray,
        ground_truth_mm: Optional[float] = None,
        condition: Optional[str] = None,
        reference_index: int = 0,
        ground_truth_presets: Optional[Sequence[float]] = None,
    ) -> SceneResult:
        """Measure every object in the frame and render them together.

        ``reference_index`` selects which object a single ``ground_truth_mm``
        refers to. Alternatively, pass ``ground_truth_presets`` to auto-match
        each object to its nearest known length (within a tolerance).

        Raises:
            ValueError: If no valid object is detected.
        """
        contours, mask = self._segment(image)
        results = [self._measure(c, mask, ground_truth_mm) for c in contours]

        ground_truths = None
        if ground_truth_presets:
            ground_truths = [match_ground_truth(r.length_mm, ground_truth_presets) for r in results]

        annotated = draw_measurements(
            image=image,
            contours=[r.contour for r in results],
            measurements=[r.measurement for r in results],
            length_mms=[r.length_mm for r in results],
            ground_truth_mm=ground_truth_mm,
            condition=condition,
            reference_index=reference_index,
            ground_truths=ground_truths,
        )
        return SceneResult(measurements=results, mask=mask, annotated_image=annotated)
