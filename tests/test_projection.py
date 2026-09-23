"""Tests for the camera projection utilities and projection-derived calibration."""

import unittest

from config.system_config import default_config
from src.projection import (
    mm_per_pixel,
    mm_per_pixel_from_camera,
    pinhole_project,
    projected_size_mm,
    projected_size_px,
    sensor_mm_to_px,
)


class TestProjection(unittest.TestCase):
    def test_pinhole_projection(self):
        x_px, y_px = pinhole_project(10.0, 5.0, 100.0, focal_length_mm=50.0)
        self.assertAlmostEqual(x_px, 5.0)
        self.assertAlmostEqual(y_px, 2.5)

    def test_pinhole_rejects_behind_camera(self):
        with self.assertRaises(ValueError):
            pinhole_project(1.0, 1.0, 0.0, focal_length_mm=50.0)

    def test_projected_size_mm(self):
        # f=50, real 10 mm at 100 mm -> 5 mm on sensor
        self.assertAlmostEqual(projected_size_mm(10.0, 100.0, 50.0), 5.0)

    def test_sensor_mm_to_px(self):
        # 18 mm on a 36 mm sensor at 1920 px -> 960 px
        self.assertAlmostEqual(sensor_mm_to_px(18.0, 36.0, 1920.0), 960.0)

    def test_projected_size_px_composition(self):
        composed = projected_size_px(10.0, 100.0, 50.0, 36.0, 1920.0)
        manual = sensor_mm_to_px(projected_size_mm(10.0, 100.0, 50.0), 36.0, 1920.0)
        self.assertAlmostEqual(composed, manual)

    def test_mm_per_pixel_inverts_projection(self):
        # A 1 mm object should span mm_per_pixel pixels.
        scale = mm_per_pixel(397.0, 8.0, 5.650, 2062)
        projected = projected_size_px(1.0, 397.0, 8.0, 5.650, 2062)
        self.assertAlmostEqual(projected, 1.0 / scale)


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
