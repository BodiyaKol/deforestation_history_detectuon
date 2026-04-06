"""
Builds spatial forest masks based on first image.

Why masks:
  SVD "learns" patterns from all images. If first images already have
  deforestation — algorithm will treat it as normal background and miss it.
  Masks allow:
    1. Track ONLY pixels that were initially forest
    2. Remove fields / roads / buildings from anomalies (not interesting)

Input:  X, H, W
Output: forest_mask, nonforest_mask  — boolean arrays (H * W)
"""

import numpy as np


# Config
NDVI_FOREST_THRESHOLD    = 0.4   # NDVI > this -> considered forest
NDVI_NONFOREST_THRESHOLD = 0.2   # NDVI < this -> definitely not forest


def build_forest_masks(
    X: np.ndarray,
    H: int,
    W: int,
    forest_threshold: float    = NDVI_FOREST_THRESHOLD,
    nonforest_threshold: float = NDVI_NONFOREST_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Builds forest and non-forest masks from FIRST image.

    Use first image as "baseline state":
      - forest_mask    : where NDVI > 0.4  -> initial forest
      - nonforest_mask : where NDVI < 0.2  -> never forest

    Parameters
    ----------
    X                   : (pixels, frames) - NDVI matrix
    H, W                : height and width of one image
    forest_threshold    : NDVI threshold for forest
    nonforest_threshold : NDVI threshold for non-forest

    Returns
    -------
    forest_mask    : (H, W) bool - initial forest pixels
    nonforest_mask : (H, W) bool - pixels that were never forest
    """
    first_frame = X[:, 0].reshape(H, W)

    forest_mask    = first_frame > forest_threshold
    nonforest_mask = first_frame < nonforest_threshold

    total          = H * W
    forest_px      = forest_mask.sum()
    nonforest_px   = nonforest_mask.sum()

    print(f"[mask] Forest in first image   : {forest_px} px  ({100 * forest_px / total:.1f}%)")
    print(f"[mask] Non-forest in first image: {nonforest_px} px  ({100 * nonforest_px / total:.1f}%)")

    return forest_mask, nonforest_mask
