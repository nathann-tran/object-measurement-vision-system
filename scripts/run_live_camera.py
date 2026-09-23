"""Live camera acquisition and measurement demonstration using Aravis and OpenCV.

Supports:
- Physical IDS camera connection via Aravis 0.8
- Sensor resolution (2472 x 2062) verification
- Automatic exposure and gain (continuous by default) for dark/light scenes
- Optional camera white balance (auto once/continuous or manual RGB ratios)
- Optional real-time measurement overlay
- Keyboard controls:
    'q' / ESC : Cleanly exit and release camera
    's'       : Save current full-resolution frame
    'm'       : Toggle real-time measurement overlay
    'e'       : Run auto exposure + gain once
    'w'       : Run auto white balance once on the current scene
"""

import argparse
from datetime import datetime
from pathlib import Path
import sys
import time

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from src.camera_interface import AravisCamera
from src.pipeline import MeasurementPipeline


def run_live_view(
    camera_id: str | None = None,
    enable_measurement: bool = False,
    display_width: int = 1280,
    save_dir: str = str(PROJECT_ROOT / "data/samples"),
    auto_exposure: str | None = "continuous",
    auto_gain: str | None = "continuous",
    exposure_time_us: float | None = None,
    gain_db: float | None = None,
    exposure_target: float = 110.0,
    white_balance: str | None = None,
    white_balance_ratios: tuple[float, float, float] | None = None,
) -> None:
    print("=" * 65)
    print("IDS INDUSTRIAL CAMERA LIVE VIEW (Aravis + OpenCV)")
    print("=" * 65)

    try:
        camera = AravisCamera(
            camera_id=camera_id,
            auto_exposure=auto_exposure,
            auto_gain=auto_gain,
            exposure_time_us=exposure_time_us,
            gain_db=gain_db,
            exposure_target=exposure_target,
            white_balance=white_balance,
            white_balance_ratios=white_balance_ratios,
        )
    except Exception as err:
        print(f"\n[Camera Error] {err}")
        print("\nTroubleshooting:")
        print("1. Ensure IDS camera is connected via USB 3.0.")
        print("2. Verify camera is visible with `arv-tool-0.8`.")
        return

    info = camera.get_info()
    exposure_us = info.get("exposure_time_us")
    exposure_txt = f"{exposure_us:.0f} us" if exposure_us is not None else "n/a"
    gain = info.get("gain_db")
    gain_txt = f"{gain:.2f} dB" if gain is not None else "n/a"
    print(f"Connected:       {info.get('model', 'Camera')} ({info.get('device_id', 'N/A')})")
    print(f"Resolution:      {info.get('width', 2472)} x {info.get('height', 2062)}")
    print(f"Exposure:        {info.get('auto_exposure') or 'n/a'} ({exposure_txt})")
    print(f"Gain:            {info.get('auto_gain') or 'n/a'} ({gain_txt})")
    print(f"White balance:   {info.get('white_balance') or 'unchanged (camera default)'}")
    print(f"Measurement:     {'ACTIVE' if enable_measurement else 'OFF (press m to toggle)'}")
    print("\nControls (click the live window first so it receives key presses):")
    print("  'q' / ESC : Exit")
    print("  's'       : Save current frame to disk")
    print("  'm'       : Toggle measurement overlay")
    print("  'e'       : Run auto exposure + gain once")
    print("  'w'       : Run auto white balance once")
    print("=" * 65)

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    pipeline = MeasurementPipeline() if enable_measurement else None
    window_name = "IDS Camera Live Feed"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    # On macOS the window must be focused to receive key presses; bring it to front.
    cv2.moveWindow(window_name, 40, 40)
    try:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
    except Exception:
        pass

    frame_count = 0
    t_start = time.time()
    fps = 0.0

    try:
        while True:
            try:
                frame = camera.get_frame()
            except TimeoutError:
                continue
            except Exception as e:
                print(f"Acquisition error: {e}")
                break

            # Host-side auto exposure/gain toward the target brightness.
            camera.update_auto_exposure(frame)

            frame_count += 1
            if frame_count % 15 == 0:
                dt = time.time() - t_start
                if dt > 0:
                    fps = 15.0 / dt
                t_start = time.time()

            display_frame = frame
            if pipeline is not None:
                try:
                    scene = pipeline.process_all(frame)
                    display_frame = scene.annotated_image if scene.annotated_image is not None else frame
                except Exception:
                    display_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    cv2.putText(
                        display_frame,
                        "Searching for nail...",
                        (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 165, 255),
                        2,
                    )
            elif display_frame.ndim == 2:
                display_frame = cv2.cvtColor(display_frame, cv2.COLOR_GRAY2BGR)

            # Overlay FPS
            cv2.putText(
                display_frame,
                f"FPS: {fps:.1f} | Frame: {frame.shape[1]}x{frame.shape[0]}",
                (20, display_frame.shape[0] - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            # Resize display copy to fit screen
            h, w = display_frame.shape[:2]
            scale = display_width / float(w)
            screen_img = cv2.resize(display_frame, (display_width, int(round(h * scale))), interpolation=cv2.INTER_AREA)

            cv2.imshow(window_name, screen_img)

            key = cv2.waitKey(10) & 0xFF
            if key in (27, ord("q")):
                break
            elif key == ord("s"):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                snap_file = save_path / f"nail_{timestamp}.png"
                cv2.imwrite(str(snap_file), frame)
                print(f"Saved snapshot: {snap_file}")
            elif key == ord("m"):
                pipeline = MeasurementPipeline() if pipeline is None else None
                status = "ENABLED" if pipeline is not None else "DISABLED"
                print(f"Measurement overlay: {status}")
            elif key == ord("e"):
                if camera.run_auto_once():
                    print("Auto exposure/gain: running once...")
                else:
                    print("Auto exposure/gain: not available on this camera.")
            elif key == ord("w"):
                if camera.set_white_balance_auto("once"):
                    print("White balance: running once...")
                else:
                    print("White balance: feature not available on this camera.")

    finally:
        camera.close()
        cv2.destroyAllWindows()
        print("Camera released. Exited cleanly.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Live camera acquisition with Aravis 0.8.")
    parser.add_argument("--camera-id", type=str, default=None, help="Camera device ID")
    parser.add_argument("--measure", action="store_true", help="Start with live measurement enabled")
    parser.add_argument("--width", type=int, default=1280, help="Display window width")
    parser.add_argument(
        "--save-dir",
        type=str,
        default=str(PROJECT_ROOT / "data/samples"),
        help="Snapshot save folder",
    )
    parser.add_argument(
        "--auto-exposure",
        choices=["off", "once", "continuous"],
        default="continuous",
        help="Camera ExposureAuto mode (default: continuous)",
    )
    parser.add_argument(
        "--auto-gain",
        choices=["off", "once", "continuous"],
        default="continuous",
        help="Camera GainAuto mode (default: continuous; use 'off' for lower noise)",
    )
    parser.add_argument("--exposure-time-us", type=float, default=None, help="Fixed exposure time in microseconds")
    parser.add_argument("--gain-db", type=float, default=None, help="Fixed gain in dB")
    parser.add_argument(
        "--exposure-target",
        type=float,
        default=110.0,
        help="Target mean brightness (0-255) for host-side auto exposure (default: 110)",
    )
    parser.add_argument(
        "--white-balance",
        choices=["off", "once", "continuous"],
        default=None,
        help="Enable camera white balance (default: leave camera setting unchanged)",
    )
    parser.add_argument("--wb-red", type=float, default=None, help="Manual red balance ratio")
    parser.add_argument("--wb-green", type=float, default=None, help="Manual green balance ratio")
    parser.add_argument("--wb-blue", type=float, default=None, help="Manual blue balance ratio")
    args = parser.parse_args()

    ratios = None
    if args.wb_red is not None and args.wb_green is not None and args.wb_blue is not None:
        ratios = (args.wb_red, args.wb_green, args.wb_blue)
    elif any(v is not None for v in (args.wb_red, args.wb_green, args.wb_blue)):
        parser.error("Manual white balance requires all of --wb-red, --wb-green and --wb-blue.")

    run_live_view(
        camera_id=args.camera_id,
        enable_measurement=args.measure,
        display_width=args.width,
        save_dir=args.save_dir,
        auto_exposure=args.auto_exposure,
        auto_gain=args.auto_gain,
        exposure_time_us=args.exposure_time_us,
        gain_db=args.gain_db,
        exposure_target=args.exposure_target,
        white_balance=args.white_balance,
        white_balance_ratios=ratios,
    )


if __name__ == "__main__":
    main()
