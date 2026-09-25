"""Live accuracy logging and summary for the demonstration.

Collects per-trial measurements (with optional ground truth and an illumination
condition label), appends them to a CSV, and produces a per-condition summary
table suitable for the accuracy section of the report.
"""

import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

_FIELDS = [
    "timestamp",
    "condition",
    "object_id",
    "ground_truth_mm",
    "measured_px",
    "measured_mm",
    "absolute_error_mm",
    "relative_error_percent",
]


def match_ground_truth(
    length_mm: float,
    presets: Sequence[float],
    max_relative_error: float = 0.15,
) -> Optional[float]:
    """Return the preset closest to ``length_mm`` within a tolerance, else ``None``.

    The tolerance is the smaller of:

    - half the gap to the nearest adjacent preset (so a measurement can't be
      matched across the midpoint between two known sizes), and
    - ``max_relative_error`` × the preset.

    This keeps auto-matching convenient while rejecting ambiguous or absurd
    matches (which would otherwise understate the measurement error).
    """
    values = sorted(float(p) for p in presets)
    if not values:
        return None

    nearest = min(values, key=lambda p: abs(p - length_mm))
    index = values.index(nearest)
    gaps = []
    if index > 0:
        gaps.append(nearest - values[index - 1])
    if index < len(values) - 1:
        gaps.append(values[index + 1] - nearest)
    half_gap = min(gaps) / 2.0 if gaps else float("inf")

    tolerance = min(half_gap, max_relative_error * nearest)
    return nearest if abs(length_mm - nearest) <= tolerance else None


class AccuracyLog:
    """Append-only accuracy log with an in-memory summary."""

    def __init__(self, csv_path: Union[str, Path]) -> None:
        self.csv_path = Path(csv_path)
        self.rows: List[Dict[str, str]] = []

    def add(
        self,
        condition: Optional[str],
        object_id: str,
        measured_px: float,
        measured_mm: float,
        ground_truth_mm: Optional[float] = None,
    ) -> Dict[str, str]:
        """Record one measurement; error columns are filled when ground truth is given."""
        abs_err: Optional[float] = None
        rel_err: Optional[float] = None
        if ground_truth_mm is not None and ground_truth_mm > 0:
            abs_err = abs(measured_mm - ground_truth_mm)
            rel_err = abs_err / ground_truth_mm * 100.0

        row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "condition": condition or "unknown",
            "object_id": object_id,
            "ground_truth_mm": f"{ground_truth_mm:.2f}" if ground_truth_mm is not None else "",
            "measured_px": f"{measured_px:.2f}",
            "measured_mm": f"{measured_mm:.3f}",
            "absolute_error_mm": f"{abs_err:.3f}" if abs_err is not None else "",
            "relative_error_percent": f"{rel_err:.2f}" if rel_err is not None else "",
        }
        self.rows.append(row)
        self._append(row)
        return row

    def _append(self, row: Dict[str, str]) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self.csv_path.exists() or self.csv_path.stat().st_size == 0
        with self.csv_path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def summary(self) -> List[Dict[str, Any]]:
        """Per-condition statistics over rows that have a ground truth."""
        groups: Dict[str, List[tuple[float, float]]] = {}
        for row in self.rows:
            if row["absolute_error_mm"] == "":
                continue
            groups.setdefault(row["condition"], []).append(
                (float(row["absolute_error_mm"]), float(row["relative_error_percent"]))
            )

        summary: List[Dict[str, Any]] = []
        for condition, values in groups.items():
            abs_vals = [a for a, _ in values]
            rel_vals = [r for _, r in values]
            summary.append(
                {
                    "condition": condition,
                    "n": len(values),
                    "mean_abs": sum(abs_vals) / len(abs_vals),
                    "mean_rel": sum(rel_vals) / len(rel_vals),
                    "max_abs": max(abs_vals),
                }
            )
        return summary

    def format_table(self) -> str:
        rows = self.summary()
        if not rows:
            return "No accuracy measurements recorded (press 'r' during the demo)."
        lines = [
            f"{'Condition':<16} | {'N':>3} | {'Mean Abs (mm)':>13} | {'Mean Rel (%)':>12} | {'Max Abs (mm)':>12}",
            "-" * 70,
        ]
        for row in rows:
            lines.append(
                f"{row['condition']:<16} | {int(row['n']):>3} | {row['mean_abs']:>13.3f} | "
                f"{row['mean_rel']:>12.2f} | {row['max_abs']:>12.3f}"
            )
        return "\n".join(lines)
