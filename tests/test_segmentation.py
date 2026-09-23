"""Tests for image segmentation and contour extraction."""

import unittest
import cv2
import numpy as np

from src.segmentation import segment_nail, segment_nails, to_grayscale


class TestSegmentation(unittest.TestCase):
    def setUp(self):
        # Create light background (220) with a dark rectangle (30)
        self.image = np.full((300, 300), 220, dtype=np.uint8)
        self.image[100:180, 50:250] = 30  # 200px wide, 80px high nail

    def test_to_grayscale(self):
        bgr = np.zeros((50, 50, 3), dtype=np.uint8)
        gray = to_grayscale(bgr)
        self.assertEqual(gray.shape, (50, 50))
        self.assertEqual(gray.ndim, 2)

    def test_segment_nail_success(self):
        contour, mask = segment_nail(self.image, blur_kernel=5, invert=True, min_area=500.0)
        self.assertEqual(mask.shape, (300, 300))
        self.assertIsNotNone(contour)
        self.assertGreater(cv2.contourArea(contour), 500.0)

    def test_otsu_method_still_available(self):
        contour, _ = segment_nail(self.image, method="otsu", min_area=500.0)
        self.assertGreater(cv2.contourArea(contour), 500.0)

    def test_segment_nail_no_object_raises(self):
        # Blank background with no object
        blank = np.full((200, 200), 220, dtype=np.uint8)
        with self.assertRaises(ValueError):
            segment_nail(blank, min_area=500.0)

    def test_segment_nails_finds_multiple_objects(self):
        canvas = np.full((300, 400), 220, dtype=np.uint8)
        canvas[60:72, 40:160] = 30  # object 1
        canvas[200:212, 240:360] = 30  # object 2
        contours, _ = segment_nails(canvas, block_size=101, min_area=200.0)
        self.assertGreaterEqual(len(contours), 2)

    def test_adaptive_survives_bright_highlight(self):
        # A strong highlight must not swallow the dark object (phone-light test).
        image = self.image.astype(np.float32)
        yy, xx = np.ogrid[:300, :300]
        spot = np.exp(-(((xx - 60) ** 2 + (yy - 60) ** 2) / (2 * 60.0 ** 2)))
        image = np.clip(image + 200.0 * spot, 0, 255).astype(np.uint8)
        contour, _ = segment_nail(image, min_area=500.0)
        self.assertGreater(cv2.contourArea(contour), 500.0)


if __name__ == "__main__":
    unittest.main()
