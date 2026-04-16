"""
patch_extractor.py — Sliding window patch extraction from Whole Slide Images.

Handles both WSI files (via OpenSlide through SlideWrapper) and standard
images (via PIL/OpenCV). Extracts tissue patches using a configurable
sliding window, skipping background regions.
"""

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from utils.config import DEFAULT_PATCH_SIZE, DEFAULT_STRIDE, DEFAULT_TISSUE_THRESHOLD
from utils.slide_utils import SlideWrapper, load_slide
from data.tissue_filter import create_tissue_mask, is_tissue_patch, is_background_patch_rgb


# ══════════════════════════════════════════════════
#  PATCH DATA CLASS
# ══════════════════════════════════════════════════

class PatchInfo:
    """Container for an extracted patch and its spatial metadata."""

    __slots__ = ["row", "col", "x", "y", "patch_size", "image"]

    def __init__(self, row, col, x, y, patch_size, image):
        self.row = row          # Grid row index
        self.col = col          # Grid column index
        self.x = x              # X coordinate (pixels) in the slide
        self.y = y              # Y coordinate (pixels) in the slide
        self.patch_size = patch_size
        self.image = image      # PIL Image or numpy array (RGB)

    def to_dict(self):
        return {
            "row": self.row, "col": self.col,
            "x": self.x, "y": self.y,
            "patch_size": self.patch_size,
        }


# ══════════════════════════════════════════════════
#  WSI PATCH EXTRACTION
# ══════════════════════════════════════════════════

def extract_patches_from_wsi(
    slide_wrapper,
    patch_size=DEFAULT_PATCH_SIZE,
    stride=DEFAULT_STRIDE,
    tissue_threshold=DEFAULT_TISSUE_THRESHOLD,
    level=0,
    thumbnail_size=2048,
    progress_callback=None,
):
    """
    Extract tissue patches from a Whole Slide Image using sliding window.

    Strategy:
      1. Generate a thumbnail at reduced resolution
      2. Create tissue mask on the thumbnail
      3. Map tissue mask coordinates back to full-resolution
      4. Extract patches at full resolution, skipping background

    Args:
        slide_wrapper: SlideWrapper instance.
        patch_size: size of each square patch in pixels.
        stride: step size of the sliding window.
        tissue_threshold: minimum tissue fraction to keep a patch.
        level: pyramid level to read patches from (0 = full resolution).
        thumbnail_size: size of thumbnail for tissue detection.
        progress_callback: optional callable(current, total) for UI progress.

    Returns:
        patches: list of PatchInfo objects.
        grid_shape: (num_rows, num_cols) of the patch grid.
        tissue_mask_thumbnail: the tissue mask at thumbnail scale (for visualization).
    """
    # Get dimensions at the target level
    level_dims = slide_wrapper.level_dimensions
    if level >= len(level_dims):
        level = len(level_dims) - 1

    slide_w, slide_h = level_dims[level]
    downsample = slide_wrapper.level_downsamples[level]

    # Generate thumbnail and tissue mask
    thumbnail = slide_wrapper.get_thumbnail((thumbnail_size, thumbnail_size))
    thumb_np = np.array(thumbnail)
    tissue_mask = create_tissue_mask(thumb_np)

    # Scale factors between thumbnail and slide level
    thumb_h, thumb_w = tissue_mask.shape[:2]
    scale_x = thumb_w / slide_w
    scale_y = thumb_h / slide_h

    # Calculate grid dimensions
    num_rows = max(1, (slide_h - patch_size) // stride + 1)
    num_cols = max(1, (slide_w - patch_size) // stride + 1)
    total_positions = num_rows * num_cols

    patches = []
    checked = 0

    for r in range(num_rows):
        for c in range(num_cols):
            y = r * stride
            x = c * stride

            # Map to thumbnail coordinates for tissue check
            tx = int(x * scale_x)
            ty = int(y * scale_y)
            tw = max(1, int(patch_size * scale_x))
            th = max(1, int(patch_size * scale_y))

            # Clip to thumbnail bounds
            tx = min(tx, thumb_w - 1)
            ty = min(ty, thumb_h - 1)
            tw = min(tw, thumb_w - tx)
            th = min(th, thumb_h - ty)

            # Check tissue content on thumbnail mask
            mask_patch = tissue_mask[ty:ty + th, tx:tx + tw]
            if not is_tissue_patch(mask_patch, tissue_threshold):
                checked += 1
                if progress_callback:
                    progress_callback(checked, total_positions)
                continue

            # Read patch from WSI at the target level
            # OpenSlide read_region uses level-0 coordinates
            level0_x = int(x * downsample)
            level0_y = int(y * downsample)

            try:
                patch_img = slide_wrapper.read_region(
                    (level0_x, level0_y), level, (patch_size, patch_size)
                )

                # Double-check: skip if the actual patch is mostly white
                if is_background_patch_rgb(patch_img):
                    checked += 1
                    if progress_callback:
                        progress_callback(checked, total_positions)
                    continue

                patches.append(PatchInfo(
                    row=r, col=c,
                    x=level0_x, y=level0_y,
                    patch_size=patch_size,
                    image=patch_img,
                ))
            except Exception as e:
                print(f"[WARNING] Failed to read patch at ({level0_x}, {level0_y}): {e}")

            checked += 1
            if progress_callback:
                progress_callback(checked, total_positions)

    return patches, (num_rows, num_cols), tissue_mask


# ══════════════════════════════════════════════════
#  STANDARD IMAGE PATCH EXTRACTION
# ══════════════════════════════════════════════════

def extract_patches_from_image(
    image_path_or_np,
    patch_size=DEFAULT_PATCH_SIZE,
    stride=DEFAULT_STRIDE,
    tissue_threshold=DEFAULT_TISSUE_THRESHOLD,
    progress_callback=None,
):
    """
    Extract tissue patches from a standard image (JPG/PNG) using sliding window.

    Args:
        image_path_or_np: file path (str) or numpy RGB array.
        patch_size: size of each square patch.
        stride: sliding window step size.
        tissue_threshold: minimum tissue fraction to keep a patch.
        progress_callback: optional callable(current, total) for UI progress.

    Returns:
        patches: list of PatchInfo objects.
        grid_shape: (num_rows, num_cols).
        tissue_mask: binary tissue mask.
    """
    # Load image
    if isinstance(image_path_or_np, str):
        image_bgr = cv2.imread(image_path_or_np)
        if image_bgr is None:
            raise FileNotFoundError(f"Cannot read image: {image_path_or_np}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    elif isinstance(image_path_or_np, np.ndarray):
        image_rgb = image_path_or_np
    else:
        raise ValueError("Expected file path or numpy array")

    h, w = image_rgb.shape[:2]

    # If image is too small, upscale it
    if h < patch_size or w < patch_size:
        scale = max(patch_size / h, patch_size / w) * 1.5
        new_w, new_h = int(w * scale), int(h * scale)
        print(f"[PATCH] Upscaling small image ({w}×{h}) → ({new_w}×{new_h})")
        image_rgb = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        h, w = new_h, new_w

    # Create tissue mask
    tissue_mask = create_tissue_mask(image_rgb)

    # Calculate grid
    num_rows = max(1, (h - patch_size) // stride + 1)
    num_cols = max(1, (w - patch_size) // stride + 1)
    total_positions = num_rows * num_cols

    patches = []
    checked = 0

    for r in range(num_rows):
        for c in range(num_cols):
            y = r * stride
            x = c * stride

            # Check tissue mask
            mask_patch = tissue_mask[y:y + patch_size, x:x + patch_size]
            if not is_tissue_patch(mask_patch, tissue_threshold):
                checked += 1
                if progress_callback:
                    progress_callback(checked, total_positions)
                continue

            # Extract RGB patch
            patch_np = image_rgb[y:y + patch_size, x:x + patch_size]
            patch_pil = Image.fromarray(patch_np)

            patches.append(PatchInfo(
                row=r, col=c,
                x=x, y=y,
                patch_size=patch_size,
                image=patch_pil,
            ))

            checked += 1
            if progress_callback:
                progress_callback(checked, total_positions)

    return patches, (num_rows, num_cols), tissue_mask


# ══════════════════════════════════════════════════
#  UNIFIED EXTRACTION INTERFACE
# ══════════════════════════════════════════════════

def extract_patches(
    slide_path,
    patch_size=DEFAULT_PATCH_SIZE,
    stride=DEFAULT_STRIDE,
    tissue_threshold=DEFAULT_TISSUE_THRESHOLD,
    level=0,
    progress_callback=None,
):
    """
    Unified patch extraction — automatically detects WSI vs standard image.

    Args:
        slide_path: path to the slide or image file.
        patch_size: patch dimensions (square).
        stride: sliding window stride.
        tissue_threshold: min tissue fraction per patch.
        level: pyramid level for WSI (ignored for standard images).
        progress_callback: optional callable(current, total).

    Returns:
        patches: list of PatchInfo objects.
        grid_shape: (num_rows, num_cols).
        tissue_mask: binary tissue mask.
        slide_wrapper: SlideWrapper (or None for standard images).
    """
    from utils.config import is_wsi_file

    slide_wrapper = None

    if is_wsi_file(slide_path):
        slide_wrapper = load_slide(slide_path)
        patches, grid_shape, tissue_mask = extract_patches_from_wsi(
            slide_wrapper,
            patch_size=patch_size,
            stride=stride,
            tissue_threshold=tissue_threshold,
            level=level,
            progress_callback=progress_callback,
        )
    else:
        patches, grid_shape, tissue_mask = extract_patches_from_image(
            slide_path,
            patch_size=patch_size,
            stride=stride,
            tissue_threshold=tissue_threshold,
            progress_callback=progress_callback,
        )

    print(f"[PATCH] Grid: {grid_shape[0]}×{grid_shape[1]} | "
          f"Tissue patches: {len(patches)} | "
          f"Patch size: {patch_size} | Stride: {stride}")

    return patches, grid_shape, tissue_mask, slide_wrapper
