"""Object measurement vision system source package.

Submodules are resolved lazily so importing a lightweight module (for example
``src.projection``) does not pull in hardware-dependent modules such as
``src.camera_interface``. This also keeps ``config`` and ``src`` free of import
cycles.
"""

import importlib

_EXPORTS = {
    # camera_interface
    "AravisCamera": "camera_interface",
    "FileFrameSource": "camera_interface",
    # segmentation
    "segment_nail": "segmentation",
    "segment_nails": "segmentation",
    "to_grayscale": "segmentation",
    # measurement
    "measure_nail": "measurement",
    "NailMeasurement": "measurement",
    # calibration
    "default_mm_per_pixel": "calibration",
    "pixels_to_mm": "calibration",
    # projection
    "mm_per_pixel": "projection",
    "mm_per_pixel_from_camera": "projection",
    # visualization
    "draw_measurement": "visualization",
    "draw_measurements": "visualization",
    "create_diagnostic_view": "visualization",
    # pipeline
    "MeasurementPipeline": "pipeline",
    "MeasurementResult": "pipeline",
    "SceneResult": "pipeline",
    # live_view
    "FrameSource": "live_view",
    "run_live_view": "live_view",
    # accuracy_log
    "AccuracyLog": "accuracy_log",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(f".{module_name}", __name__)
    return getattr(module, name)
