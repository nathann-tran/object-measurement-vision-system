"""Image segmentation and contour extraction using standard OpenCV operations.

Implements the conventional industrial vision workflow:
Grayscale -> Gaussian Blur -> Threshold -> Small Morphology (Opening) -> Contour Selection.

Two thresholding strategies are available:

- ``"adaptive"`` (default): local adaptive thresholding. Robust to strong and
  non-uniform illumination (ring light, bright highlights, shadows), where a
  single global threshold would separate the wrong region.
- ``"otsu"``: global Otsu thresholding for evenly lit scenes.
"""

from typing import List, Tuple
import cv2
import numpy as np

from config.system_config import default_config


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Ensure image is an 8-bit single-channel 2D array."""
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image.astype(np.uint8)


def _threshold(
    blurred: np.ndarray,
    method: str,
    invert: bool,
    block_size: int,
    c: float,
) -> np.ndarray:
    """Binarize a blurred grayscale image; foreground objects become 255."""
    if method == "adaptive":
        if block_size % 2 == 0:
            block_size += 1
        thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        return cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresh_type,
            block_size,
            c,
        )

    thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, mask = cv2.threshold(blurred, 0, 255, thresh_type + cv2.THRESH_OTSU)
    return mask


def segment_nails(
    image: np.ndarray,
    blur_kernel: int = default_config.segmentation.gaussian_blur_kernel,
    method: str = default_config.segmentation.threshold_method,
    invert: bool = default_config.segmentation.invert_threshold,
    block_size: int = default_config.segmentation.adaptive_block_size,
    c: float = default_config.segmentation.adaptive_c,
    morphology_kernel_size: int = default_config.segmentation.morphology_kernel_size,
    min_area: float = default_config.segmentation.min_object_area_px,
    max_area_fraction: float = default_config.segmentation.max_object_area_fraction,
    max_extent_fraction: float = default_config.segmentation.max_object_extent_fraction,
    max_objects: int = default_config.segmentation.max_objects,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """Segment all nail/object silhouettes from the background.

    Returns:
        tuple[list[np.ndarray], np.ndarray]: (nail_contours largest-first, binary_mask)

    Raises:
        ValueError: If no valid foreground object with area >= min_area is detected.
    """
    gray = to_grayscale(image)

    # 1. Gaussian blur to suppress sensor noise
    blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)

    # 2. Thresholding (adaptive local threshold or global Otsu)
    mask = _threshold(blurred, method, invert, block_size, c)

    # 3. Small morphological opening to remove isolated noise
    kernel_size = max(1, morphology_kernel_size)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 4. Contour extraction
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 5. Object selection: keep valid contours, reject tiny noise, whole-frame
    frame_h, frame_w = image.shape[0], image.shape[1]
    total_area = float(frame_h * frame_w)
    max_area = max_area_fraction * total_area
    max_w = max_extent_fraction * frame_w
    max_h = max_extent_fraction * frame_h

    def _filter(c: np.ndarray) -> bool:
        area = cv2.contourArea(c)
        if not (min_area <= area <= max_area):
            return False
        # Reject illumination artifacts that span (nearly) the whole frame.
        _, _, w, h = cv2.boundingRect(c)
        return w <= max_w and h <= max_h

    valid_contours = [c for c in contours if _filter(c)]
    if not valid_contours:
        raise ValueError(f"No object detected with area between {min_area} and {max_area} pixels.")

    valid_contours.sort(key=cv2.contourArea, reverse=True)
    return valid_contours[:max_objects], mask


def segment_nail(
    image: np.ndarray,
    blur_kernel: int = default_config.segmentation.gaussian_blur_kernel,
    method: str = default_config.segmentation.threshold_method,
    invert: bool = default_config.segmentation.invert_threshold,
    block_size: int = default_config.segmentation.adaptive_block_size,
    c: float = default_config.segmentation.adaptive_c,
    morphology_kernel_size: int = default_config.segmentation.morphology_kernel_size,
    min_area: float = default_config.segmentation.min_object_area_px,
    max_area_fraction: float = default_config.segmentation.max_object_area_fraction,
    max_extent_fraction: float = default_config.segmentation.max_object_extent_fraction,
) -> tuple[np.ndarray, np.ndarray]:
    """Segment the single largest nail from the background and return its contour.

    Thin wrapper around :func:`segment_nails` kept for single-object callers.
    """
    contours, mask = segment_nails(
        image,
        blur_kernel=blur_kernel,
        method=method,
        invert=invert,
        block_size=block_size,
        c=c,
        morphology_kernel_size=morphology_kernel_size,
        min_area=min_area,
        max_area_fraction=max_area_fraction,
        max_extent_fraction=max_extent_fraction,
        max_objects=1,
    )
    return contours[0], mask
