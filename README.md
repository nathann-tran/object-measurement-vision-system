# Object Measurement Vision System — MVP

An industrial computer-vision system for measuring the physical longitudinal length of nails in millimeters from images. Designed for an IDS industrial camera (2472 × 2062 sensor resolution, 8 mm lens, 397 mm working distance) with Aravis acquisition.

Refactored into a concise, explainable, and conventional industrial vision pipeline using standard OpenCV primitives (`cv2.threshold` Otsu, `cv2.findContours`, `cv2.minAreaRect`, `cv2.morphologyEx`), Aravis 0.8 industrial camera acquisition (with optional white balance), and projection-derived pixel-to-mm calibration.

---

## 1. System Architecture & Pipeline Flow

The system strictly decouples camera acquisition from image processing, measurement geometry, calibration, and visualization:

```text
┌────────────────────────────────────────────────────────┐
│ 1. Frame Acquisition Layer (src/camera_interface.py)   │
│    - AravisCamera (hardware acquisition via Aravis 0.8)│
│    - FileFrameSource (offline image file testing)      │
└───────────────────────────┬────────────────────────────┘
                            │ (2062, 2472) uint8 BGR/Gray frame
                            ▼
┌────────────────────────────────────────────────────────┐
│ 2. Preprocessing & Segmentation (src/segmentation.py)  │
│    - Grayscale conversion (cv2.cvtColor)               │
│    - Noise suppression (cv2.GaussianBlur 5×5)          │
│    - Global bimodal separation (cv2.threshold Otsu)    │
│    - Speckle removal (cv2.morphologyEx MORPH_OPEN 3×3) │
└───────────────────────────┬────────────────────────────┘
                            │ binary mask (foreground = 255)
                            ▼
┌────────────────────────────────────────────────────────┐
│ 3. Object Selection (src/segmentation.py)              │
│    - cv2.findContours (RETR_EXTERNAL, CHAIN_APPROX_SIMP│
│    - Area filtering (min_area to 90% image area)       │
│    - Selection of primary nail silhouette contour      │
└───────────────────────────┬────────────────────────────┘
                            │ primary contour points
                            ▼
┌────────────────────────────────────────────────────────┐
│ 4. Geometric Measurement (src/measurement.py)          │
│    - cv2.minAreaRect(contour) minimum-area rectangle   │
│    - Longitudinal length = max(width, height) in px    │
│    - Rotation angle and short-edge midpoint endpoints  │
└───────────────────────────┬────────────────────────────┘
                            │ length_px, endpoints, angle
                            ▼
┌────────────────────────────────────────────────────────┐
│ 5. Calibration (src/calibration.py)                    │
│    - Projection scale: mm_per_pixel = 0.13597 mm/px    │
│    - Calibrated length_mm = length_px * mm_per_pixel   │
└───────────────────────────┬────────────────────────────┘
                            │ calibrated dimension in mm
                            ▼
┌────────────────────────────────────────────────────────┐
│ 6. Output & Evaluation (src/visualization.py)          │
│    - Visual overlay: contour, minAreaRect, HUD banner  │
│    - Measurement logging: data/results/measurements.csv│
└────────────────────────────────────────────────────────┘
```

---

## 2. Mathematical Principles & Code Walkthrough

### Why `cv2.minAreaRect`?
Nails can be placed at arbitrary orientations ($0^\circ$ to $90^\circ$). Simple axis-aligned bounding boxes change their dimensions drastically when rotated:
$$\text{bbox\_w} = L \cos\theta + W \sin\theta, \quad \text{bbox\_h} = L \sin\theta + W \cos\theta$$

Neither represents true physical length. `cv2.minAreaRect(contour)` computes the minimum rotated bounding rectangle enclosing the contour points using rotating calipers:
- **Rotation-invariant**: Measures the true longitudinal extent regardless of the object's orientation in the field of view.
- **Translation-invariant**: Depends strictly on relative contour boundary coordinates.
- **Direct endpoints**: Yields the midpoints of the two short opposing edges, corresponding to the physical contact points of a mechanical caliper on the nail cap and tip.

### Otsu Thresholding
For controlled industrial backlighting or front illumination, the image histogram exhibits a bimodal distribution (dark nail silhouette against light background, or vice versa). Otsu's method calculates the optimal threshold $T^*$ that maximizes between-class variance:

$$\sigma_b^2(T) = \omega_0(T) \omega_1(T) [\mu_0(T) - \mu_1(T)]^2$$

Inverted thresholding (`cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU`) ensures the dark nail foreground is cleanly mapped to pixel value 255.

### Morphological Cleanup
A small $3 \times 3$ elliptical kernel is applied with morphological opening (`cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)`). This cleans speckles and dust particles while preserving the exact geometric boundaries and sharp tips of the nail.

### Pixel-to-Millimeter Calibration
The default scale is derived from the camera projection model in
`docs/projection-note.md` (implemented in `src/projection.py`). For a 1 mm object
at the working distance:

$$\text{mm\_per\_pixel} = \frac{\text{real\_distance} \times \text{sensor\_size\_mm}}{\text{focal\_length} \times \text{sensor\_size\_px}}$$

$$\text{Length}_{\text{mm}} = L_{\text{pixels}} \times \text{mm\_per\_pixel}$$

For our optical setup (working distance $397$ mm, lens focal length $8.0$ mm,
sensor $6.773 \times 5.650$ mm, resolution $2472 \times 2062$, pixel pitch
$2.74\ \mu\text{m}$) the projection scale is **$0.13597\ \text{mm/pixel}$**
(medium 30.0 mm nail = 220.6 px). The sensor dimensions are consistent with the
pixel pitch ($6.773 / 2472 = 5.650 / 2062 = 2.74\ \mu\text{m}$). An empirical
override measured from a known reference is available via
`src.calibration.calculate_scale` and can be set on `CalibrationConfig`.

---

## 3. Environment Setup

The project uses Python virtual environments managed via `uv` with `pyproject.toml`.

```bash
# 1. Create virtual environment linked to Homebrew system site-packages (for Aravis and PyGObject)
uv venv --python /opt/homebrew/bin/python3 --system-site-packages

# 2. Activate virtual environment
source .venv/bin/activate

# 3. Install the project in editable mode so `import config` and `import src`
#    resolve from any working directory (including the VS Code run/debug button)
uv pip install -e .
```

> If you prefer not to install, run commands from the project root. The root
> `conftest.py` and the `sys.path` bootstrap in each script also keep imports
> working for tests and scripts.

---

## 4. Running the System

### Step 1: Run Automated Unit and Integration Tests
```bash
python3 -m unittest discover tests -v
```
All 20 unit and integration tests verify:
- Grayscale conversion and Otsu segmentation on synthetic & file sources.
- No-object error handling and background noise rejection.
- Minimum area rectangle dimension accuracy and rotation invariance ($0^\circ$ to $90^\circ$).
- Pixel-to-mm conversions, empirical scaling, and the projection-derived scale.
- Full end-to-end pipeline execution distinguishing small (24 mm), medium (30 mm), and large (55 mm) nails.

### Step 2: Measure a Single Image
```bash
python3 scripts/run_measurement.py data/synthetic/medium/medium_rot_30deg.png --ground-truth 30.0 --output data/results/medium_rot_30deg_result.png
```
Console output:
```text
Measured Length:   29.86 mm (219.6 pixels)
Measured Width:    3.66 mm
Rotation Angle:    -62.1 degrees
Ground Truth:      30.00 mm
Absolute Error:    0.14 mm
Relative Error:    0.47 %
```

### Step 3: Run the Benchmark Dataset & Accuracy Evaluation
```bash
python3 scripts/evaluate_dataset.py --save-images data/results
```
Evaluates all benchmark conditions (39 images across sizes, rotations, positions, and lighting) and writes the results to `data/results/measurements.csv`. The three `gradient` samples are rejected by global Otsu segmentation (known limitation, see Section 5).

### Step 4: Connect and Activate the Camera

1. Connect the IDS camera with a USB 3.0 cable to a USB 3.0 port (usually blue) and let the OS enumerate it.
2. Confirm the host sees the device and inspect its GenICam feature tree:
   ```bash
   arv-tool-0.8                 # list connected Aravis / GenICam devices
   arv-tool-0.8 -n <device-id>  # dump the camera's features
   ```
   If no device appears, re-seat the cable/port and confirm the camera has power.
3. Run the live view (the editable install from Section 3 keeps imports working):
   ```bash
   # Basic live stream
   python3 scripts/run_live_camera.py

   # Live stream with real-time measurement overlay
   python3 scripts/run_live_camera.py --measure

   # Live stream with camera white balance enabled
   python3 scripts/run_live_camera.py --white-balance continuous
   ```
   Auto exposure and auto gain default to `continuous`, so the view adapts to dark/bright scenes. This IDS camera exposes no `ExposureAuto`/`GainAuto`, so the script performs auto exposure on the host (adjusting `ExposureTime`, then `Gain`). Override with `--auto-exposure`, `--auto-gain`, `--exposure-time-us`, `--gain-db`. Use `--camera-id <device-id>` when several cameras are connected.

Interactive keyboard controls in the OpenCV window:
- `q` or `ESC`: Exit cleanly and release stream and camera hardware.
- `s`: Save the current raw frame snapshot to `data/samples/`.
- `m`: Toggle measurement overlay on/off dynamically.
- `e`: Run auto exposure + gain once on the current scene.
- `w`: Run auto white balance once on the current scene.

Exposure / white balance options:
- `--auto-exposure off|once|continuous` and `--auto-gain off|once|continuous` (default `continuous`).
- `--exposure-time-us <us>` / `--gain-db <dB>`: fixed values (disable the matching auto mode).
- `--white-balance off|once|continuous`: sets the GenICam `BalanceWhiteAuto` mode (default leaves the camera setting unchanged).
- `--wb-red/--wb-green/--wb-blue <float>`: sets manual `BalanceRatio` gains (all three required).

See [`docs/run-live-camera.md`](docs/run-live-camera.md) for the full camera guide (detection, options, controls, troubleshooting).

---

## 5. Accuracy Evaluation & Results

The system logs all measurements to `data/results/measurements.csv` as mandated by Section 10 of the refactor specification.

### Benchmark Summary Across Illumination Conditions:

| Illumination Condition | Samples Measured | Success Rate | Mean Absolute Error (mm) | Mean Relative Error (%) | Max Error (mm) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Normal** (Various Angles/Positions) | 30 / 30 | 100% | **0.150 mm** | **0.46%** | 0.358 mm |
| **Bright** (Overexposed) | 3 / 3 | 100% | **0.159 mm** | **0.48%** | 0.197 mm |
| **Dim** (Underexposed) | 3 / 3 | 100% | **0.159 mm** | **0.48%** | 0.197 mm |
| **Gradient** (Non-uniform) | 0 / 3 | 0%* | N/A | N/A | N/A |
| **Overall Valid Set** | **36 / 39** | **92%** | **0.151 mm** | **0.46%** | **0.358 mm** |

*\*Note on Gradient condition*: Under a strong illumination ramp the histogram is dominated by the background, so global Otsu splits the background and yields a frame-spanning region. Object selection rejects contours whose bounding box spans ≥90% of the frame (`max_object_extent_fraction`), so these samples are skipped rather than mis-measured. Per the refactor spec, adaptive/CLAHE preprocessing is intentionally omitted until real-camera testing shows it is needed.

---

## 6. Project Structure

```text
object-measurement-vision-system/
├── config/
│   ├── __init__.py
│   └── system_config.py      # CameraConfig, NailGroundTruth, CalibrationConfig, SegmentationConfig
├── src/
│   ├── __init__.py           # Lazy package exports
│   ├── camera_interface.py   # AravisCamera (Aravis 0.8, white balance) & FileFrameSource
│   ├── projection.py         # Pinhole projection & sensor mm -> px (docs/projection-note.md)
│   ├── segmentation.py       # Grayscale, blur, Otsu threshold, morphology, contour selection
│   ├── measurement.py        # cv2.minAreaRect geometry, length_px, caliper endpoints, angle
│   ├── calibration.py        # PixelCalibration & projection/empirical scaling math
│   ├── pipeline.py           # Standard pipeline orchestrator returning MeasurementResult
│   └── visualization.py      # HUD overlay, contour drawing, caliper line, bounding box
├── scripts/
│   ├── run_live_camera.py    # Live streaming, keyboard shortcuts, white balance, measurement toggle
│   ├── run_measurement.py    # CLI single-image measurement
│   ├── evaluate_dataset.py   # Benchmark evaluation script writing to measurements.csv
│   └── generate_synthetic_data.py # Synthetic image generator for offline unit testing
├── tests/
│   ├── test_segmentation.py  # Tests for segmentation & contour extraction
│   ├── test_measurement.py   # Tests for minAreaRect & rotation invariance
│   ├── test_calibration.py   # Tests for mm/pixel conversions
│   ├── test_projection.py    # Tests for projection math & derived calibration
│   └── test_pipeline.py      # End-to-end integration tests
├── conftest.py               # Adds project root to sys.path for tests
├── data/
│   ├── results/
│   │   └── measurements.csv  # Required measurement accuracy log
│   └── samples/              # Saved snapshots from live camera
├── docs/
│   ├── refactor-spec.md      # Refactor specification
│   ├── projection-note.md    # Camera projection foundations used by calibration
│   ├── run-live-camera.md    # Guide: connect and run the Aravis camera live
│   ├── spec.md               # Original system specification
│   └── build-poc.md          # PoC guidelines
├── pyproject.toml            # Project dependencies and metadata
└── README.md                 # System documentation & walkthrough
```
