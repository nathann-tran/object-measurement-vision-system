"""Synthetic nail image dataset generator.

Generates realistic 2D images of nails with mathematically verified dimensions
for testing measurement accuracy under controlled variations:
- Object dimensions: 24.0 mm (small), 30.0 mm (medium), 55.0 mm (large)
- Rotations: 0, 15, 30, 45, 60, 90 degrees
- Positions: Center, Left, Right, Top, Bottom
- Illumination: Normal, Bright, Dim, Gradient (non-uniform)
- Noise: Gaussian sensor noise and subtle edge antialiasing
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from config.system_config import default_config
from src.calibration import default_mm_per_pixel


def render_synthetic_nail(
    length_mm: float,
    rotation_deg: float,
    center_pos: tuple[int, int],
    canvas_shape: tuple[int, int] = (2062, 2472),
    mm_per_pixel: float = default_mm_per_pixel(),
    illumination: str = "normal",
    add_noise: bool = True,
) -> np.ndarray:
    """Render a synthetic nail on an image canvas.

    Geometry:
    - Head: flat cap with width ~2.2x shaft diameter, thickness ~1.0 mm
    - Shaft: cylindrical body
    - Tip: tapered triangular point with length ~2.5 mm
    - Total span along axis equals length_mm.

    Args:
        length_mm: Ground-truth total length in mm.
        rotation_deg: Rotation angle in degrees.
        center_pos: (cx, cy) pixel coordinates on canvas.
        canvas_shape: (height, width) of generated image.
        mm_per_pixel: Calibration scale factor.
        illumination: "normal", "bright", "dim", or "gradient".
        add_noise: Whether to inject Gaussian noise.

    Returns:
        np.ndarray: Grayscale synthetic image (uint8).
    """
    canvas_h, canvas_w = canvas_shape
    total_length_px = length_mm / mm_per_pixel

    # Physical proportions in mm converted to pixels
    head_thick_px = 1.0 / mm_per_pixel
    head_diam_px = 3.5 / mm_per_pixel
    shaft_diam_px = 1.6 / mm_per_pixel
    tip_length_px = 2.5 / mm_per_pixel

    shaft_length_px = total_length_px - head_thick_px - tip_length_px
    if shaft_length_px < 0:
        raise ValueError(f"Nail length {length_mm} mm is too short for nail anatomy model.")

    # 1. Construct nail polygon in local coordinates (aligned with X axis, origin at head edge)
    # Head outermost surface is at x = -total_length_px / 2
    x_head_outer = -total_length_px / 2.0
    x_head_inner = x_head_outer + head_thick_px
    x_shaft_end = x_head_inner + shaft_length_px
    x_tip_apex = total_length_px / 2.0  # x_shaft_end + tip_length_px

    y_head_top = head_diam_px / 2.0
    y_head_bottom = -head_diam_px / 2.0
    y_shaft_top = shaft_diam_px / 2.0
    y_shaft_bottom = -shaft_diam_px / 2.0

    # Polygon points tracing the full perimeter
    local_poly = np.array(
        [
            [x_head_outer, y_head_bottom],
            [x_head_outer, y_head_top],
            [x_head_inner, y_head_top],
            [x_head_inner, y_shaft_top],
            [x_shaft_end, y_shaft_top],
            [x_tip_apex, 0.0],  # Sharp tip
            [x_shaft_end, y_shaft_bottom],
            [x_head_inner, y_shaft_bottom],
            [x_head_inner, y_head_bottom],
        ],
        dtype=np.float32,
    )

    # 2. Rotate and translate polygon
    theta = np.radians(rotation_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    rot_matrix = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)

    transformed_poly = (local_poly @ rot_matrix.T) + np.array(center_pos, dtype=np.float32)

    # 3. Create background intensity based on illumination condition
    if illumination == "bright":
        base_bg = 245.0
        nail_val = 60.0
    elif illumination == "dim":
        base_bg = 150.0
        nail_val = 30.0
    elif illumination == "gradient":
        # Horizontal illumination gradient across image
        grad_x = np.linspace(160.0, 240.0, canvas_w, dtype=np.float32)
        base_bg = np.tile(grad_x, (canvas_h, 1))
        nail_val = 45.0
    else:  # "normal"
        base_bg = 220.0
        nail_val = 40.0

    if isinstance(base_bg, float):
        canvas = np.full((canvas_h, canvas_w), base_bg, dtype=np.float32)
    else:
        canvas = base_bg.copy()

    # 4. Antialiased rendering: draw filled polygon on mask
    mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    poly_int = np.round(transformed_poly).astype(np.int32)
    cv2.fillPoly(mask, [poly_int], 255, lineType=cv2.LINE_AA)

    # Smooth the boundary slightly for realistic optical blur
    mask_smooth = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (3, 3), 0.8)

    # Blend foreground and background
    canvas = canvas * (1.0 - mask_smooth) + nail_val * mask_smooth

    # 5. Sensor noise
    if add_noise:
        noise = np.random.normal(0, 3.5, canvas.shape).astype(np.float32)
        canvas = canvas + noise

    # Clip and convert to uint8
    image = np.clip(canvas, 0, 255).astype(np.uint8)
    return image


def generate_poc_dataset(
    output_dir: Path | str,
    mm_per_pixel: float = default_mm_per_pixel(),
    canvas_shape: tuple[int, int] = (2062, 2472),
) -> List[Dict]:
    """Generate the full benchmark dataset covering all required experimental conditions."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    h, w = canvas_shape
    cx, cy = w // 2, h // 2

    # Defined ground truths per spec
    nail_sizes = [
        ("small", default_config.ground_truth.small_mm),
        ("medium", default_config.ground_truth.medium_mm),
        ("large", default_config.ground_truth.large_mm),
    ]

    # Variations
    rotations = [0.0, 15.0, 30.0, 45.0, 60.0, 90.0]
    positions = {
        "center": (cx, cy),
        "left": (cx - 450, cy),
        "right": (cx + 450, cy),
        "top": (cx, cy - 350),
        "bottom": (cx, cy + 350),
    }
    illuminations = ["normal", "bright", "dim", "gradient"]

    manifest = []
    total_count = 0

    print(f"Generating synthetic PoC dataset in: {out_path} ...")

    # 1. Base rotation suite for each size
    for size_name, length_mm in nail_sizes:
        size_dir = out_path / size_name
        size_dir.mkdir(exist_ok=True)

        # Vary rotation at center under normal illumination
        for rot in rotations:
            filename = f"{size_name}_rot_{int(rot):02d}deg.png"
            filepath = size_dir / filename
            img = render_synthetic_nail(
                length_mm=length_mm,
                rotation_deg=rot,
                center_pos=positions["center"],
                canvas_shape=canvas_shape,
                mm_per_pixel=mm_per_pixel,
                illumination="normal",
            )
            cv2.imwrite(str(filepath), img)

            entry = {
                "filename": str(filepath.relative_to(out_path)),
                "object_name": f"{size_name.capitalize()} nail",
                "ground_truth_mm": length_mm,
                "rotation_deg": rot,
                "position": "center",
                "illumination": "normal",
                "category": "rotation_variation",
                "mm_per_pixel": mm_per_pixel,
            }
            manifest.append(entry)
            total_count += 1

        # Vary lateral position for medium nail (and others)
        for pos_name, pos_coords in positions.items():
            if pos_name == "center":
                continue  # Already generated in rotation suite
            filename = f"{size_name}_pos_{pos_name}.png"
            filepath = size_dir / filename
            img = render_synthetic_nail(
                length_mm=length_mm,
                rotation_deg=15.0,  # Realistic slight slant
                center_pos=pos_coords,
                canvas_shape=canvas_shape,
                mm_per_pixel=mm_per_pixel,
                illumination="normal",
            )
            cv2.imwrite(str(filepath), img)

            entry = {
                "filename": str(filepath.relative_to(out_path)),
                "object_name": f"{size_name.capitalize()} nail",
                "ground_truth_mm": length_mm,
                "rotation_deg": 15.0,
                "position": pos_name,
                "illumination": "normal",
                "category": "position_variation",
                "mm_per_pixel": mm_per_pixel,
            }
            manifest.append(entry)
            total_count += 1

        # Vary illumination for each size
        for illum in illuminations:
            if illum == "normal":
                continue  # Already generated
            filename = f"{size_name}_illum_{illum}.png"
            filepath = size_dir / filename
            img = render_synthetic_nail(
                length_mm=length_mm,
                rotation_deg=20.0,
                center_pos=positions["center"],
                canvas_shape=canvas_shape,
                mm_per_pixel=mm_per_pixel,
                illumination=illum,
            )
            cv2.imwrite(str(filepath), img)

            entry = {
                "filename": str(filepath.relative_to(out_path)),
                "object_name": f"{size_name.capitalize()} nail",
                "ground_truth_mm": length_mm,
                "rotation_deg": 20.0,
                "position": "center",
                "illumination": illum,
                "category": "illumination_variation",
                "mm_per_pixel": mm_per_pixel,
            }
            manifest.append(entry)
            total_count += 1

    # Save manifest
    manifest_path = out_path / "dataset_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Generated {total_count} images. Manifest saved to {manifest_path}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic nail benchmark dataset.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "data/synthetic"),
        help="Target folder for dataset generation",
    )
    args = parser.parse_args()
    generate_poc_dataset(args.output_dir)


if __name__ == "__main__":
    main()
