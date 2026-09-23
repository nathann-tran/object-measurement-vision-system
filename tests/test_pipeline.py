"""Integration tests for the complete measurement pipeline."""

import unittest
import numpy as np

from scripts.generate_synthetic_data import render_synthetic_nail
from src.pipeline import MeasurementPipeline


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.scale = 0.136
        self.pipeline = MeasurementPipeline(mm_per_pixel=self.scale)

    def test_pipeline_on_medium_nail(self):
        gt_mm = 30.0
        img = render_synthetic_nail(
            length_mm=gt_mm,
            rotation_deg=25.0,
            center_pos=(1236, 1031),
            canvas_shape=(2062, 2472),
            mm_per_pixel=self.scale,
            illumination="normal",
            add_noise=False,
        )

        result = self.pipeline.process(img, ground_truth_mm=gt_mm)

        self.assertIsNotNone(result.contour)
        self.assertIsNotNone(result.mask)
        self.assertIsNotNone(result.annotated_image)

        # Measured length should be close to ground truth (within 0.6 mm)
        self.assertAlmostEqual(result.length_mm, gt_mm, delta=0.6)
        self.assertIsNotNone(result.absolute_error_mm)
        self.assertLess(result.relative_error_percent or 0.0, 2.5)

    def test_pipeline_distinguishes_sizes(self):
        lengths: dict[float, float] = {}
        for size_mm in [24.0, 30.0, 55.0]:
            img = render_synthetic_nail(
                length_mm=size_mm,
                rotation_deg=0.0,
                center_pos=(1236, 1031),
                canvas_shape=(2062, 2472),
                mm_per_pixel=self.scale,
                illumination="normal",
                add_noise=False,
            )
            res = self.pipeline.process(img, ground_truth_mm=size_mm)
            lengths[size_mm] = res.length_mm

        # Ordering must strictly be 24 < 30 < 55
        self.assertLess(lengths[24.0], lengths[30.0])
        self.assertLess(lengths[30.0], lengths[55.0])

    def test_process_all_detects_multiple_nails(self):
        img1 = render_synthetic_nail(
            length_mm=30.0,
            rotation_deg=0.0,
            center_pos=(700, 1031),
            canvas_shape=(2062, 2472),
            mm_per_pixel=self.scale,
            illumination="normal",
            add_noise=False,
        )
        img2 = render_synthetic_nail(
            length_mm=55.0,
            rotation_deg=30.0,
            center_pos=(1700, 1031),
            canvas_shape=(2062, 2472),
            mm_per_pixel=self.scale,
            illumination="normal",
            add_noise=False,
        )
        combined = np.minimum(img1, img2)

        scene = self.pipeline.process_all(combined)

        self.assertGreaterEqual(scene.count, 2)
        self.assertIsNotNone(scene.annotated_image)


if __name__ == "__main__":
    unittest.main()
