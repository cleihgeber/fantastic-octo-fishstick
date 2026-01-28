"""Image preprocessing for raster to vector conversion."""

import numpy as np
import cv2
from pathlib import Path
from typing import Union, Tuple, Optional

from .config import ConversionConfig, ThresholdMethod


def load_image(
    source: Union[str, Path, np.ndarray],
) -> np.ndarray:
    """Load an image from file or use provided array.

    Args:
        source: File path or numpy array

    Returns:
        Image as numpy array (BGR or grayscale)

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If image cannot be loaded
    """
    if isinstance(source, np.ndarray):
        return source.copy()

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not load image: {path}")

    return image


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert image to grayscale.

    Args:
        image: Input image (BGR, BGRA, or grayscale)

    Returns:
        Grayscale image
    """
    if len(image.shape) == 2:
        return image

    if image.shape[2] == 4:
        # BGRA - handle transparency
        bgr = image[:, :, :3]
        alpha = image[:, :, 3] / 255.0

        # Composite onto white background
        white_bg = np.ones_like(bgr) * 255
        composite = (bgr * alpha[:, :, np.newaxis] + white_bg * (1 - alpha[:, :, np.newaxis])).astype(
            np.uint8
        )
        return cv2.cvtColor(composite, cv2.COLOR_BGR2GRAY)

    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def denoise(image: np.ndarray, strength: int = 10) -> np.ndarray:
    """Apply denoising to reduce artifacts.

    Args:
        image: Grayscale image
        strength: Denoising strength (higher = more smoothing)

    Returns:
        Denoised image
    """
    return cv2.fastNlMeansDenoising(image, None, strength, 7, 21)


def binarize(
    image: np.ndarray,
    method: ThresholdMethod = ThresholdMethod.OTSU,
    threshold_value: int = 128,
    invert: bool = False,
    blur_kernel: int = 3,
) -> np.ndarray:
    """Convert grayscale image to binary (black and white).

    Args:
        image: Grayscale image
        method: Thresholding method to use
        threshold_value: Fixed threshold value (for FIXED method)
        invert: Invert the result (for white-on-dark images)
        blur_kernel: Gaussian blur kernel size (0 = no blur)

    Returns:
        Binary image (0 and 255 values)
    """
    # Apply slight blur to reduce noise
    if blur_kernel > 0:
        if blur_kernel % 2 == 0:
            blur_kernel += 1
        image = cv2.GaussianBlur(image, (blur_kernel, blur_kernel), 0)

    if method == ThresholdMethod.OTSU:
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == ThresholdMethod.ADAPTIVE:
        binary = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
    else:  # FIXED
        _, binary = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)

    if invert:
        binary = cv2.bitwise_not(binary)

    return binary


def clean_binary(
    binary: np.ndarray, min_object_size: int = 10, close_gaps: bool = True
) -> np.ndarray:
    """Clean up binary image by removing small objects and closing gaps.

    Args:
        binary: Binary image
        min_object_size: Remove connected components smaller than this
        close_gaps: Apply morphological closing to connect nearby lines

    Returns:
        Cleaned binary image
    """
    # Ensure lines are black (0) on white (255) background for consistency
    # We'll work with inverted image where lines are white
    inverted = cv2.bitwise_not(binary)

    if close_gaps:
        # Small closing to connect nearby line segments
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        inverted = cv2.morphologyEx(inverted, cv2.MORPH_CLOSE, kernel)

    # Remove small objects
    if min_object_size > 0:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inverted, connectivity=8)

        # Create mask of components to keep
        keep_mask = np.zeros_like(inverted)
        for i in range(1, num_labels):  # Skip background (0)
            if stats[i, cv2.CC_STAT_AREA] >= min_object_size:
                keep_mask[labels == i] = 255

        inverted = keep_mask

    # Return to original convention (black lines on white)
    return cv2.bitwise_not(inverted)


def preprocess(
    source: Union[str, Path, np.ndarray], config: ConversionConfig
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Complete preprocessing pipeline.

    Args:
        source: Image source (path or array)
        config: Conversion configuration

    Returns:
        Tuple of (binary image with black lines on white background, original dimensions)
    """
    # Load and convert to grayscale
    image = load_image(source)
    original_size = (image.shape[1], image.shape[0])  # (width, height)

    gray = to_grayscale(image)

    # Optional denoising
    if config.denoise:
        gray = denoise(gray)

    # Binarize
    binary = binarize(
        gray,
        method=config.threshold_method,
        threshold_value=config.threshold_value,
        invert=config.invert,
        blur_kernel=config.blur_kernel,
    )

    # Clean up
    binary = clean_binary(binary, min_object_size=config.remove_small_objects)

    return binary, original_size
