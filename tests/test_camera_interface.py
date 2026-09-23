"""Tests for Aravis camera feature control (no hardware required)."""

import unittest
from typing import Any, cast

import numpy as np

from src.camera_interface import AravisCamera


class FakeCamera:
    """Minimal stand-in exposing the Aravis feature accessors we use."""

    def __init__(self, available: bool = True) -> None:
        self.available = available
        self.strings: dict[str, str] = {}
        self.floats: dict[str, float] = {}
        self.bounds: dict[str, tuple[float, float]] = {
            "ExposureTime": (10.0, 100000.0),
            "Gain": (1.0, 13.0),
        }

    def is_feature_available(self, name: str) -> bool:
        return self.available

    def set_string(self, name: str, value: str) -> None:
        self.strings[name] = value

    def get_string(self, name: str) -> str | None:
        return self.strings.get(name)

    def set_float(self, name: str, value: float) -> None:
        self.floats[name] = value

    def get_float(self, name: str) -> float:
        return self.floats.get(name, 1000.0)

    def get_float_bounds(self, name: str) -> tuple[float, float]:
        return self.bounds[name]


def make_camera(available: bool = True, with_state: bool = False) -> tuple[AravisCamera, FakeCamera]:
    camera = object.__new__(AravisCamera)
    fake = FakeCamera(available=available)
    camera.camera = cast(Any, fake)
    if with_state:
        camera.exposure_target = 110.0
        camera.exposure_tolerance = 12.0
        camera.auto_exposure_interval = 1
        camera._auto_exposure_mode = None
        camera._auto_gain_mode = None
        camera._exposure_bounds = (10.0, 100000.0)
        camera._gain_bounds = (1.0, 13.0)
        camera._exposure_time = 1000.0
        camera._gain = 1.0
        camera._ae_frame_count = 0
    return camera, fake


class TestWhiteBalance(unittest.TestCase):
    def test_set_auto_mode(self):
        camera, _ = make_camera()
        self.assertTrue(camera.set_white_balance_auto("continuous"))
        self.assertEqual(camera.get_white_balance_mode(), "Continuous")

    def test_invalid_mode_raises(self):
        camera, _ = make_camera()
        with self.assertRaises(ValueError):
            camera.set_white_balance_auto("bogus")

    def test_manual_ratios_disable_auto(self):
        camera, fake = make_camera()
        self.assertTrue(camera.set_white_balance_ratios((1.5, 1.0, 2.0)))
        self.assertEqual(fake.strings["BalanceWhiteAuto"], "Off")
        self.assertEqual(fake.floats["BalanceRatio"], 2.0)

    def test_no_settings_returns_false(self):
        camera, _ = make_camera()
        self.assertFalse(camera.configure_white_balance())

    def test_unavailable_feature_returns_false(self):
        camera, _ = make_camera(available=False)
        self.assertFalse(camera.set_white_balance_auto("once"))


class TestExposureCameraSide(unittest.TestCase):
    def test_set_auto_exposure(self):
        camera, fake = make_camera()
        self.assertTrue(camera.set_auto_exposure("continuous"))
        self.assertEqual(fake.strings["ExposureAuto"], "Continuous")

    def test_set_auto_gain(self):
        camera, fake = make_camera()
        self.assertTrue(camera.set_auto_gain("once"))
        self.assertEqual(fake.strings["GainAuto"], "Once")

    def test_invalid_auto_mode_raises(self):
        camera, _ = make_camera()
        with self.assertRaises(ValueError):
            camera.set_auto_exposure("bogus")


class TestExposureManual(unittest.TestCase):
    def test_set_exposure_time(self):
        camera, fake = make_camera()
        self.assertTrue(camera.set_exposure_time(1000.0))
        self.assertEqual(fake.floats["ExposureTime"], 1000.0)

    def test_set_gain(self):
        camera, fake = make_camera()
        self.assertTrue(camera.set_gain(3.0))
        self.assertEqual(fake.floats["Gain"], 3.0)


class TestSoftwareAutoExposure(unittest.TestCase):
    def test_dark_frame_increases_exposure(self):
        camera, fake = make_camera(available=False, with_state=True)
        self.assertTrue(camera.enable_software_auto_exposure("continuous"))
        camera.update_auto_exposure(np.zeros((10, 10), dtype=np.uint8))
        self.assertGreater(fake.floats["ExposureTime"], 1000.0)

    def test_bright_frame_decreases_exposure(self):
        camera, fake = make_camera(available=False, with_state=True)
        camera.enable_software_auto_exposure("continuous")
        camera.update_auto_exposure(np.full((10, 10), 255, dtype=np.uint8))
        self.assertLess(fake.floats["ExposureTime"], 1000.0)

    def test_on_target_frame_does_not_change_exposure(self):
        camera, fake = make_camera(available=False, with_state=True)
        camera.enable_software_auto_exposure("continuous")
        camera.update_auto_exposure(np.full((10, 10), 110, dtype=np.uint8))
        self.assertNotIn("ExposureTime", fake.floats)

    def test_once_mode_stops_when_on_target(self):
        camera, _ = make_camera(available=False, with_state=True)
        camera.enable_software_auto_exposure("once")
        camera.update_auto_exposure(np.full((10, 10), 110, dtype=np.uint8))
        self.assertEqual(camera._auto_exposure_mode, "off")

    def test_gain_used_when_exposure_saturated(self):
        camera, fake = make_camera(available=False, with_state=True)
        camera._exposure_time = 100000.0  # already at max
        camera.enable_software_auto_exposure("continuous")
        camera.enable_software_auto_gain("continuous")
        camera.update_auto_exposure(np.zeros((10, 10), dtype=np.uint8))
        self.assertGreater(fake.floats["Gain"], 1.0)

    def test_configure_falls_back_to_software(self):
        camera, _ = make_camera(available=False, with_state=True)
        self.assertTrue(camera.configure_exposure(auto_exposure="continuous"))
        self.assertEqual(camera._auto_exposure_mode, "continuous")


if __name__ == "__main__":
    unittest.main()
