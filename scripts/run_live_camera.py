"""Live camera launcher: configure the Aravis camera, then start the live view.

This script owns all camera/hardware configuration (device selection, exposure,
gain, white balance). The application/entry logic (display loop, measurement
overlay, keyboard controls) lives in :mod:`src.live_view` and is intentionally
free of camera-specific settings, so it can be explained on its own.

Usage:
    python3 scripts/run_live_camera.py [--measure] [camera options]
"""

import argparse
from pathlib import Path
import sys

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.camera_interface import AravisCamera
from src.live_view import run_live_view


def _print_camera_info(camera: AravisCamera) -> None:
    info = camera.get_info()
    exposure_us = info.get("exposure_time_us")
    exposure_txt = f"{exposure_us:.0f} us" if exposure_us is not None else "n/a"
    gain = info.get("gain_db")
    gain_txt = f"{gain:.2f} dB" if gain is not None else "n/a"
    print(f"Camera:          {info.get('model', 'Camera')} ({info.get('device_id', 'N/A')})")
    print(f"Resolution:      {info.get('width', 2472)} x {info.get('height', 2062)}")
    print(f"Exposure:        {info.get('auto_exposure') or 'n/a'} ({exposure_txt})")
    print(f"Gain:            {info.get('auto_gain') or 'n/a'} ({gain_txt})")
    print(f"White balance:   {info.get('white_balance') or 'unchanged (camera default)'}")


def _white_balance_ratios(parser: argparse.ArgumentParser, args: argparse.Namespace):
    values = (args.wb_red, args.wb_green, args.wb_blue)
    if all(v is not None for v in values):
        return values
    if any(v is not None for v in values):
        parser.error("Manual white balance requires all of --wb-red, --wb-green and --wb-blue.")
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Configure the IDS/Aravis camera and run the live measurement view."
    )

    camera_group = parser.add_argument_group("camera settings")
    camera_group.add_argument("--camera-id", type=str, default=None, help="Camera device ID")
    camera_group.add_argument(
        "--auto-exposure",
        choices=["off", "once", "continuous"],
        default="continuous",
        help="Camera ExposureAuto mode (default: continuous)",
    )
    camera_group.add_argument(
        "--auto-gain",
        choices=["off", "once", "continuous"],
        default="continuous",
        help="Camera GainAuto mode (default: continuous; use 'off' for lower noise)",
    )
    camera_group.add_argument("--exposure-time-us", type=float, default=None, help="Fixed exposure time in microseconds")
    camera_group.add_argument("--gain-db", type=float, default=None, help="Fixed gain in dB")
    camera_group.add_argument(
        "--exposure-target",
        type=float,
        default=110.0,
        help="Target mean brightness (0-255) for host-side auto exposure (default: 110)",
    )
    camera_group.add_argument(
        "--white-balance",
        choices=["off", "once", "continuous"],
        default=None,
        help="Camera BalanceWhiteAuto mode (default: leave camera setting unchanged)",
    )
    camera_group.add_argument("--wb-red", type=float, default=None, help="Manual red balance ratio")
    camera_group.add_argument("--wb-green", type=float, default=None, help="Manual green balance ratio")
    camera_group.add_argument("--wb-blue", type=float, default=None, help="Manual blue balance ratio")

    app_group = parser.add_argument_group("application settings")
    app_group.add_argument("--measure", action="store_true", help="Start with the measurement overlay enabled")
    app_group.add_argument("--width", type=int, default=1280, help="Display window width")
    app_group.add_argument(
        "--save-dir",
        type=str,
        default=str(PROJECT_ROOT / "data/samples"),
        help="Snapshot save folder",
    )
    app_group.add_argument("--ground-truth", type=float, default=None, help="Initial ground-truth length in mm (default: first preset)")
    app_group.add_argument("--condition", type=str, default=None, help="Initial illumination-condition label")
    app_group.add_argument(
        "--auto-match",
        action="store_true",
        help="Auto-match each nail to its nearest ground-truth preset (within a tolerance)",
    )
    app_group.add_argument(
        "--log-csv",
        type=str,
        default=str(PROJECT_ROOT / "data/results/live_measurements.csv"),
        help="CSV file for recorded accuracy trials",
    )

    args = parser.parse_args()
    ratios = _white_balance_ratios(parser, args)

    try:
        camera = AravisCamera(
            camera_id=args.camera_id,
            auto_exposure=args.auto_exposure,
            auto_gain=args.auto_gain,
            exposure_time_us=args.exposure_time_us,
            gain_db=args.gain_db,
            exposure_target=args.exposure_target,
            white_balance=args.white_balance,
            white_balance_ratios=ratios,
        )
    except Exception as err:
        print(f"\n[Camera Error] {err}")
        print("\nTroubleshooting:")
        print("1. Ensure IDS camera is connected via USB 3.0.")
        print("2. Verify camera is visible with `arv-tool-0.8`.")
        return

    _print_camera_info(camera)
    run_live_view(
        camera,
        enable_measurement=args.measure,
        display_width=args.width,
        save_dir=args.save_dir,
        window_name="IDS Camera Live Feed",
        initial_ground_truth=args.ground_truth,
        initial_condition=args.condition,
        auto_match=args.auto_match,
        log_csv=args.log_csv,
    )


if __name__ == "__main__":
    main()
