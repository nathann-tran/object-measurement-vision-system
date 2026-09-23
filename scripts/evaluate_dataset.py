"""Benchmark evaluation script recording accuracy across experimental variations.

Evaluates test images, calculates measurement errors, logs all runs to
data/results/measurements.csv, and prints a performance summary table.
"""

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import sys

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from src.camera_interface import FileFrameSource
from src.pipeline import MeasurementPipeline


def evaluate_dataset(
    manifest_path: str = str(PROJECT_ROOT / "data/synthetic/dataset_manifest.json"),
    csv_output_path: str = str(PROJECT_ROOT / "data/results/measurements.csv"),
    save_images_dir: str | None = None,
) -> None:
    manifest_file = Path(manifest_path)
    if not manifest_file.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_file}")

    dataset_root = manifest_file.parent
    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    csv_path = Path(csv_output_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    save_dir = Path(save_images_dir) if save_images_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    pipeline = MeasurementPipeline()

    rows = []
    condition_stats: dict[str, list[float]] = {}
    size_stats: dict[str, list[float]] = {}

    print(f"\nEvaluating {len(manifest)} test images...")
    print("-" * 75)

    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for item in manifest:
        img_rel = item["filename"]
        img_path = dataset_root / img_rel
        gt_mm = float(item["ground_truth_mm"])
        obj_name = item.get("object_name", "Nail")
        illum = item.get("illumination", "normal")
        rot = item.get("rotation_deg", 0.0)
        pos = item.get("position", "center")

        cond_label = f"illum={illum}, rot={rot:.0f}deg, pos={pos}"

        try:
            frame = FileFrameSource(img_path).get_frame()
            result = pipeline.process(frame, ground_truth_mm=gt_mm, condition=cond_label)

            abs_err = result.absolute_error_mm if result.absolute_error_mm is not None else 0.0
            rel_err = result.relative_error_percent if result.relative_error_percent is not None else 0.0

            rows.append({
                "timestamp": timestamp_str,
                "illumination_condition": illum,
                "object_id": obj_name,
                "ground_truth_mm": f"{gt_mm:.2f}",
                "measured_px": f"{result.length_px:.2f}",
                "measured_mm": f"{result.length_mm:.2f}",
                "absolute_error_mm": f"{abs_err:.3f}",
                "relative_error_percent": f"{rel_err:.2f}",
            })

            condition_stats.setdefault(illum, []).append(abs_err)
            size_stats.setdefault(obj_name, []).append(abs_err)

            if save_dir:
                out_img_path = save_dir / Path(img_rel).name
                cv2.imwrite(str(out_img_path), result.annotated_image)

        except Exception as e:
            rows.append({
                "timestamp": timestamp_str,
                "illumination_condition": illum,
                "object_id": obj_name,
                "ground_truth_mm": f"{gt_mm:.2f}",
                "measured_px": "N/A",
                "measured_mm": "N/A",
                "absolute_error_mm": "N/A",
                "relative_error_percent": "N/A",
            })
            print(f"Skipping {img_rel} (unsegmented under {illum}): {e}")

    # Write measurements to CSV
    csv_headers = [
        "timestamp",
        "illumination_condition",
        "object_id",
        "ground_truth_mm",
        "measured_px",
        "measured_mm",
        "absolute_error_mm",
        "relative_error_percent",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        writer.writeheader()
        writer.writerows(rows)

    successful_rows = [r for r in rows if r["absolute_error_mm"] != "N/A"]
    all_errors = [float(r["absolute_error_mm"]) for r in successful_rows]
    all_rel_errors = [float(r["relative_error_percent"]) for r in successful_rows]

    print("\n" + "=" * 75)
    print("OBJECT MEASUREMENT MVP — EVALUATION SUMMARY")
    print("=" * 75)
    print(f"Total Evaluated Images:       {len(rows)}")
    print(f"Successfully Measured:        {len(successful_rows)} / {len(rows)}")
    if all_errors:
        print(f"Mean Absolute Error:          {np.mean(all_errors):.3f} mm")
        print(f"Max Absolute Error:           {np.max(all_errors):.3f} mm")
        print(f"Mean Relative Error:          {np.mean(all_rel_errors):.2f} %")
        print(f"Max Relative Error:           {np.max(all_rel_errors):.2f} %")
    print(f"Results Logged To:            {csv_path}")
    print("=" * 75)

    print("\nACCURACY BY ILLUMINATION CONDITION:")
    print("-" * 65)
    print(f"{'Illumination':<18} | {'Measured':<8} | {'Mean Error (mm)':<16} | {'Max Error (mm)':<16}")
    print("-" * 65)
    for illum_name, errs in condition_stats.items():
        print(f"{illum_name:<18} | {len(errs):<8} | {np.mean(errs):<16.3f} | {np.max(errs):<16.3f}")
    print("-" * 65)

    print("\nACCURACY BY NAIL OBJECT:")
    print("-" * 65)
    print(f"{'Object':<18} | {'Measured':<8} | {'Mean Error (mm)':<16} | {'Max Error (mm)':<16}")
    print("-" * 65)
    for obj, errs in size_stats.items():
        print(f"{obj:<18} | {len(errs):<8} | {np.mean(errs):<16.3f} | {np.max(errs):<16.3f}")
    print("=" * 65)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate accuracy across conditions.")
    parser.add_argument(
        "--manifest",
        type=str,
        default=str(PROJECT_ROOT / "data/synthetic/dataset_manifest.json"),
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=str(PROJECT_ROOT / "data/results/measurements.csv"),
    )
    parser.add_argument("--save-images", type=str, default=None, help="Folder to save annotated images")
    args = parser.parse_args()

    evaluate_dataset(args.manifest, args.csv, args.save_images)


if __name__ == "__main__":
    main()
