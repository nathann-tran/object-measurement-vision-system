"""Tests for the camera-agnostic live view rendering logic."""

import unittest
from typing import Any, cast

import numpy as np

from src.live_view import _render_frame
from src.pipeline import SceneResult


class FakePipeline:
    def __init__(self, result: SceneResult | None = None, raise_error: bool = False) -> None:
        self.result = result
        self.raise_error = raise_error

    def process_all(
        self,
        frame: np.ndarray,
        ground_truth_mm: float | None = None,
        condition: str | None = None,
        reference_index: int = 0,
        ground_truth_presets=None,
    ) -> SceneResult:
        if self.raise_error:
            raise ValueError("no object detected")
        assert self.result is not None
        return self.result


class TestRenderFrame(unittest.TestCase):
    def test_without_pipeline_returns_bgr(self):
        frame = np.zeros((10, 10), dtype=np.uint8)
        image, scene = _render_frame(frame, None)
        self.assertEqual(image.shape, (10, 10, 3))
        self.assertIsNone(scene)

    def test_with_annotated_scene_returns_overlay_and_scene(self):
        frame = np.zeros((10, 10), dtype=np.uint8)
        annotated = np.full((10, 10, 3), 7, dtype=np.uint8)
        scene = SceneResult(annotated_image=annotated)
        pipeline = cast(Any, FakePipeline(scene))
        image, returned_scene = _render_frame(frame, pipeline)
        self.assertIs(image, annotated)
        self.assertIs(returned_scene, scene)

    def test_pipeline_error_shows_status_hint(self):
        frame = np.zeros((120, 500), dtype=np.uint8)
        pipeline = cast(Any, FakePipeline(raise_error=True))
        image, scene = _render_frame(frame, pipeline)
        self.assertEqual(image.shape, (120, 500, 3))
        self.assertIsNone(scene)
        self.assertGreater(int(image.sum()), 0)  # "Searching..." text was drawn


if __name__ == "__main__":
    unittest.main()
