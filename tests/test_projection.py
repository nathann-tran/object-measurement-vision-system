"""Tests for the projection-derived calibration scale."""

import unittest

from config.system_config import default_config
from src.projection import mm_per_pixel, mm_per_pixel_from_camera


class TestProjection(unittest.TestCase):
    def test_mm_per_pixel_formula(self):
        # mm_per_pixel = distance * sensor_mm / (f * sensor_px)
        expected = 397.0 * 5.650 / (8.0 * 2062)
        self.assertAlmostEqual(mm_per_pixel(397.0, 8.0, 5.650, 2062), expected, places=12)

    def test_mm_per_pixel_rejects_bad_inputs(self):
        with self.assertRaises(ValueError):
            mm_per_pixel(397.0, 0.0, 5.650, 2062)
        with self.assertRaises(ValueError):
            mm_per_pixel(397.0, 8.0, 5.650, 0)


class TestProjectionCalibration(unittest.TestCase):
    def test_scale_matches_pixel_pitch_derivation(self):
        # sensor_mm / sensor_px is the 2.74 um pixel pitch, so the note's
        # formula reduces to working_distance * pixel_pitch / focal_length.
        expected = 397.0 * (2.74 / 1000.0) / 8.0
        derived = mm_per_pixel_from_camera(8.0, 397.0, 6.773, 5.650, 2472, 2062)
        self.assertAlmostEqual(derived, expected, delta=1e-5)

    def test_default_config_uses_projection_scale(self):
        camera = default_config.camera
        calibration_scale = default_config.calibration.mm_per_pixel
        assert calibration_scale is not None
        self.assertAlmostEqual(calibration_scale, camera.mm_per_pixel, places=12)
        self.assertAlmostEqual(calibration_scale, 0.13597, delta=1e-4)

    def test_explicit_calibration_override_is_respected(self):
        from config.system_config import CalibrationConfig, SystemConfig

        config = SystemConfig(calibration=CalibrationConfig(mm_per_pixel=0.1))
        override = config.calibration.mm_per_pixel
        assert override is not None
        self.assertAlmostEqual(override, 0.1)


if __name__ == "__main__":
    unittest.main()
