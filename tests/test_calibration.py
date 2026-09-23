"""Tests for pixel-to-millimeter calibration."""

import unittest

from src.calibration import pixels_to_mm, mm_to_pixels, calculate_scale, PixelCalibration


class TestCalibration(unittest.TestCase):
    def test_conversions(self):
        scale = 0.1  # 0.1 mm per pixel
        self.assertAlmostEqual(pixels_to_mm(100.0, scale), 10.0)
        self.assertAlmostEqual(mm_to_pixels(10.0, scale), 100.0)

    def test_calculate_scale(self):
        # 30.0 mm reference measured as 300 pixels -> 0.1 mm/pixel
        scale = calculate_scale(known_length_mm=30.0, measured_pixels=300.0)
        self.assertAlmostEqual(scale, 0.1)

    def test_pixel_calibration_class(self):
        cal = PixelCalibration(mm_per_pixel=0.05)
        self.assertAlmostEqual(cal.to_mm(200.0), 10.0)
        self.assertAlmostEqual(cal.to_pixels(10.0), 200.0)

    def test_invalid_scale_raises(self):
        with self.assertRaises(ValueError):
            PixelCalibration(mm_per_pixel=0.0)


if __name__ == "__main__":
    unittest.main()
