"""
tissue_filter.py — Background removal and tissue detection in histopathology images.

Uses grayscale conversion, Otsu thresholding, and morphological operations
to create a binary tissue mask that distinguishes tissue from white background.
"""

import cv2
import numpy as np
from PIL import Image

from utils.config import DEFAULT_TISSUE_THRESHOLD


# ══════════════════════════════════════════════════
#  TISSUE MASK CREATION
# ══════════════════════════════════════════════════

def create_tissue_mask(image, kernel_size=15, morph_iterations=2):
    """
    Create a binary tissue mask by removing white/light background.

    Pipeline:
      1. Convert to grayscale
      2. Apply Otsu's thresholding (inverted: tissue = foreground)
      3. Morphological close (fill holes in tissue)
      4. Morphological open (remove small noise)

    Args:
        image: input image as numpy array (RGB or BGR) or PIL Image.
        kernel_size: size of the morphological structuring element.
        morph_iterations: number of morphological operation iterations.

    Returns:
        tissue_mask: binary mask (255 = tissue, 0 = background), numpy uint8.
    """
    # Convert PIL to numpy if needed
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Ensure BGR for OpenCV operations
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # Otsu's thresholding (inverted so tissue = white in mask)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=morph_iterations)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=morph_iterations)

    return mask


def create_tissue_mask_hsv(image, sat_threshold=15, val_threshold=240):
    """
    Alternative tissue mask using HSV color space.

    Better for slides with colored backgrounds or staining artifacts.
    Tissue regions have measurable saturation; background is low-saturation white.

    Args:
        image: input RGB image as numpy array or PIL Image.
        sat_threshold: minimum saturation to count as tissue.
        val_threshold: maximum value to count as tissue (excludes bright white).

    Returns:
        tissue_mask: binary mask (255 = tissue, 0 = background).
    """
    if isinstance(image, Image.Image):
        image = np.array(image)

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Tissue has noticeable saturation and isn't pure white
    sat_mask = hsv[:, :, 1] > sat_threshold
    val_mask = hsv[:, :, 2] < val_threshold

    tissue = (sat_mask & val_mask).astype(np.uint8) * 255

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    tissue = cv2.morphologyEx(tissue, cv2.MORPH_CLOSE, kernel, iterations=2)
    tissue = cv2.morphologyEx(tissue, cv2.MORPH_OPEN, kernel, iterations=1)

    return tissue


# ══════════════════════════════════════════════════
#  PATCH-LEVEL TISSUE CHECKS
# ══════════════════════════════════════════════════

def get_tissue_percentage(patch_mask):
    """
    Compute the fraction of tissue pixels in a patch mask.

    Args:
        patch_mask: binary mask region (255 = tissue).

    Returns:
        float between 0.0 and 1.0.
    """
    if patch_mask.size == 0:
        return 0.0
    return np.count_nonzero(patch_mask) / patch_mask.size


def is_tissue_patch(patch_mask, threshold=DEFAULT_TISSUE_THRESHOLD):
    """
    Check whether a patch mask contains enough tissue to be worth analysing.

    Args:
        patch_mask: binary mask region for the patch.
        threshold: minimum tissue fraction required.

    Returns:
        True if the patch has sufficient tissue.
    """
    return get_tissue_percentage(patch_mask) >= threshold


def is_background_patch_rgb(patch_rgb, white_threshold=220, white_fraction=0.7):
    """
    Quick check on the raw RGB patch (without a pre-computed mask).

    A patch is considered background if most pixels are near-white.

    Args:
        patch_rgb: RGB image patch as numpy array, shape (H, W, 3).
        white_threshold: pixel intensity above this is "white".
        white_fraction: fraction of near-white pixels to declare "background".

    Returns:
        True if the patch is mostly background (white).
    """
    if isinstance(patch_rgb, Image.Image):
        patch_rgb = np.array(patch_rgb)

    gray = np.mean(patch_rgb, axis=2)
    white_ratio = np.sum(gray > white_threshold) / gray.size
    return white_ratio >= white_fraction


# ══════════════════════════════════════════════════
#  TISSUE REGION STATISTICS
# ══════════════════════════════════════════════════

def get_tissue_stats(tissue_mask):
    """
    Compute statistics about the tissue mask.

    Args:
        tissue_mask: binary mask (255 = tissue).

    Returns:
        dict with tissue area stats.
    """
    total_pixels = tissue_mask.size
    tissue_pixels = np.count_nonzero(tissue_mask)
    background_pixels = total_pixels - tissue_pixels

    return {
        "total_pixels": total_pixels,
        "tissue_pixels": tissue_pixels,
        "background_pixels": background_pixels,
        "tissue_fraction": tissue_pixels / total_pixels if total_pixels > 0 else 0.0,
        "tissue_area_pct": f"{tissue_pixels / total_pixels * 100:.1f}%",
    }
