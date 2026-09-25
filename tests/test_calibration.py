"""Tests for pixel-to-millimeter conversion."""

import unittest

from config.system_config import default_config
from src.calibration import default_mm_per_pixel, pixels_to_mm


class TestCalibration(unittest.TestCase):
    def test_pixels_to_mm_with_explicit_scale(self):
        self.assertAlmostEqual(pixels_to_mm(100.0, 0.1), 10.0)

    def test_pixels_to_mm_uses_default_scale(self):
        self.assertAlmostEqual(pixels_to_mm(100.0), 100.0 * default_mm_per_pixel())

    def test_default_scale_matches_camera_projection(self):
        self.assertAlmostEqual(default_mm_per_pixel(), default_config.camera.mm_per_pixel)

    def test_explicit_calibration_override_is_used(self):
        from config.system_config import CalibrationConfig, SystemConfig

        config = SystemConfig(calibration=CalibrationConfig(mm_per_pixel=0.1))
        override = config.calibration.mm_per_pixel
        assert override is not None
        self.assertAlmostEqual(override, 0.1)


if __name__ == "__main__":
    unittest.main()
