"""Measure nail length from an image file.

Usage:
    python3 scripts/run_measurement.py path/to/image.png [--ground-truth 30.0] [--output result.png]
"""

import argparse
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure nail dimensions in millimeters from an image.")
    parser.add_argument("image_path", type=str, help="Path to input image file")
    parser.add_argument("--ground-truth", type=float, default=None, help="Caliper ground truth in mm")
    parser.add_argument("--output", type=str, default=None, help="Save annotated image to path")
    parser.add_argument("--show", action="store_true", help="Display result in OpenCV window")
    args = parser.parse_args()

    image_path = Path(args.image_path)
    if not image_path.exists():
        print(f"Error: File not found: {image_path}")
        return

    # Load image
    source = FileFrameSource(image_path)
    frame = source.get_frame()

    # Process frame
    pipeline = MeasurementPipeline()
    result = pipeline.process(frame, ground_truth_mm=args.ground_truth)

    print("=" * 60)
    print("OBJECT MEASUREMENT RESULT")
    print("=" * 60)
    print(f"Image:             {image_path.name}")
    print(f"Measured Length:   {result.length_mm:.2f} mm ({result.length_px:.1f} pixels)")
    print(f"Measured Width:    {result.width_mm:.2f} mm")
    print(f"Rotation Angle:    {result.measurement.angle_deg:.1f} degrees")
    print(f"Caliper Endpoint1: ({result.measurement.endpoint1[0]:.1f}, {result.measurement.endpoint1[1]:.1f})")
    print(f"Caliper Endpoint2: ({result.measurement.endpoint2[0]:.1f}, {result.measurement.endpoint2[1]:.1f})")

    if result.ground_truth_mm is not None:
        print("-" * 60)
        print(f"Ground Truth:      {result.ground_truth_mm:.2f} mm")
        print(f"Absolute Error:    {result.absolute_error_mm:.2f} mm")
        print(f"Relative Error:    {result.relative_error_percent:.2f} %")
    print("=" * 60)

    if args.output and result.annotated_image is not None:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), result.annotated_image)
        print(f"Saved annotated result to: {out_path}")

    if args.show and result.annotated_image is not None:
        cv2.imshow("Measurement Result", result.annotated_image)
        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
