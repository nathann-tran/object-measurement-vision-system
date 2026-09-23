"""Tests for geometric measurement using cv2.minAreaRect."""

import unittest
import numpy as np

from src.measurement import measure_nail


class TestMeasurement(unittest.TestCase):
    def test_horizontal_rectangle_dimensions(self):
        # 200 x 20 rectangle
        # Corners: (100, 100), (300, 100), (300, 120), (100, 120)
        pts = np.array([
            [100, 100],
            [300, 100],
            [300, 120],
            [100, 120],
        ], dtype=np.int32)

        meas = measure_nail(pts)
        self.assertAlmostEqual(meas.length_px, 200.0, delta=1.5)
        self.assertAlmostEqual(meas.width_px, 20.0, delta=1.5)

        # Distance between endpoints should equal length_px
        ep_dist = np.linalg.norm(np.array(meas.endpoint1) - np.array(meas.endpoint2))
        self.assertAlmostEqual(ep_dist, meas.length_px, delta=1.0)

    def test_rotated_rectangle_invariance(self):
        # Caliper length must remain constant regardless of rotation
        true_len = 220.0
        true_w = 15.0

        for angle in [0.0, 30.0, 45.0, 60.0, 90.0]:
            theta = np.radians(angle)
            rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

            # Local corners
            corners = np.array([
                [-true_len / 2, -true_w / 2],
                [true_len / 2, -true_w / 2],
                [true_len / 2, true_w / 2],
                [-true_len / 2, true_w / 2],
            ])

            corners_rot = (corners @ rot.T) + np.array([500.0, 500.0])
            corners_int = np.int32(np.round(corners_rot))

            meas = measure_nail(corners_int)
            self.assertAlmostEqual(
                meas.length_px,
                true_len,
                delta=2.0,
                msg=f"Length changed at angle {angle} degrees",
            )


if __name__ == "__main__":
    unittest.main()
