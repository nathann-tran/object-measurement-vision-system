"""Tests for the live accuracy log and summary."""

import csv
import tempfile
import unittest
from pathlib import Path
from typing import Any

from src.accuracy_log import AccuracyLog, match_ground_truth


class TestAccuracyLog(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.csv_path = Path(self._tmp.name) / "live_measurements.csv"
        self.log = AccuracyLog(self.csv_path)

    def tearDown(self):
        self._tmp.cleanup()

    def test_error_computed_when_ground_truth_given(self):
        row = self.log.add(condition="ring_light", object_id="reference", measured_px=400.0, measured_mm=54.0, ground_truth_mm=55.0)
        self.assertAlmostEqual(float(row["absolute_error_mm"]), 1.0, places=3)
        self.assertAlmostEqual(float(row["relative_error_percent"]), 1.818, places=2)

    def test_no_error_without_ground_truth(self):
        row = self.log.add(condition="normal", object_id="reference", measured_px=400.0, measured_mm=54.0)
        self.assertEqual(row["absolute_error_mm"], "")
        self.assertEqual(row["relative_error_percent"], "")

    def test_summary_groups_by_condition(self):
        self.log.add("ring_light", "reference", 400.0, 54.0, 55.0)
        self.log.add("ring_light", "reference", 401.0, 54.5, 55.0)
        self.log.add("dim", "reference", 200.0, 29.0, 30.0)
        self.log.add("dim", "reference", 200.0, 29.5)  # no ground truth -> excluded

        summary: dict[str, Any] = {str(row["condition"]): row for row in self.log.summary()}
        self.assertEqual(int(summary["ring_light"]["n"]), 2)
        self.assertEqual(int(summary["dim"]["n"]), 1)
        self.assertAlmostEqual(summary["dim"]["mean_abs"], 1.0, places=3)

    def test_csv_written_with_header_once(self):
        self.log.add("normal", "reference", 100.0, 10.0, 10.0)
        self.log.add("normal", "reference", 100.0, 10.0, 10.0)
        with self.csv_path.open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["condition"], "normal")

    def test_format_table_empty(self):
        self.assertIn("No accuracy measurements", self.log.format_table())


class TestMatchGroundTruth(unittest.TestCase):
    presets = (24.0, 30.0, 55.0)

    def test_matches_nearest_within_tolerance(self):
        self.assertEqual(match_ground_truth(29.8, self.presets), 30.0)
        self.assertEqual(match_ground_truth(54.0, self.presets), 55.0)

    def test_rejects_ambiguous_value(self):
        # 40 mm is 10 mm from 30 (tolerance 3 mm) and 15 mm from 55 -> no match.
        self.assertIsNone(match_ground_truth(40.0, self.presets))

    def test_rejects_value_beyond_half_gap(self):
        # 28 mm is 2 mm from 30, but 4 mm from 24; half-gap guard for 30 is 3 mm.
        # 28 is within 2 mm of 30, so it matches 30.
        self.assertEqual(match_ground_truth(28.0, self.presets), 30.0)
        # 33 mm is 3 mm from 30 (tolerance 3) -> boundary match.
        self.assertEqual(match_ground_truth(33.0, self.presets), 30.0)
        # 34 mm is 4 mm from 30 -> beyond tolerance -> no match.
        self.assertIsNone(match_ground_truth(34.0, self.presets))

    def test_empty_presets(self):
        self.assertIsNone(match_ground_truth(30.0, ()))


if __name__ == "__main__":
    unittest.main()
