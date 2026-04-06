"""
Spatial filtering of anomalies - removes point noise,
keeps only solid connected patches (real deforestation).

Idea:
  anomaly_mask has True where Z-score < threshold.
  But clouds, shadows, noise give isolated pixels.
  Real deforestation - SOLID patch of dozens/hundreds of pixels.

  Algorithm:
    1. For each frame unfold anomaly_mask to 2D
    2. scipy.label() finds all connected components (pixel groups)
    3. Discard components smaller than MIN_CLUSTER_SIZE
    4. Result - only large solid patches
"""

import numpy as np
from scipy.ndimage import label, binary_dilation


# Config
MIN_CLUSTER_SIZE = 10    # minimum pixels in connected group
DILATION_ITERS   = 1     # expand mask before clustering
                         # (connects almost-solid patches)


def filter_spatial_noise(
    anomaly_mask: np.ndarray,
    H: int,
    W: int,
    min_cluster_size: int = MIN_CLUSTER_SIZE,
    dilation_iters:   int = DILATION_ITERS,
) -> tuple[np.ndarray, dict]:
    """
    Removes isolated pixels from anomaly_mask.

    Parameters
    anomaly_mask     : (pixels, frames) bool
    H, W             : dimensions of one image
    min_cluster_size : minimum pixels in group (smaller - noise)
    dilation_iters   : how many times to expand mask before clustering

    Returns
    -------
    filtered_mask : (pixels, frames) bool - only large patches
    stats         : dict with info per frame
    """
    n_frames      = anomaly_mask.shape[1]
    filtered_mask = np.zeros_like(anomaly_mask)
    stats         = {"clusters_per_frame": [], "removed_noise_px": []}

    # structural element - 8-connectivity (diagonals count too)
    struct = np.ones((3, 3), dtype=bool)

    for t in range(n_frames):
        frame_mask = anomaly_mask[:, t].reshape(H, W)

        # Expansion - connects pixels that almost touch
        if dilation_iters > 0:
            dilated = binary_dilation(frame_mask, iterations=dilation_iters)
        else:
            dilated = frame_mask.copy()

        # Clustering connected components
        labeled, n_components = label(dilated, structure=struct)

        # Keep only large clusters
        clean_frame   = np.zeros((H, W), dtype=bool)
        valid_clusters = 0

        for cluster_id in range(1, n_components + 1):
            cluster_pixels = (labeled == cluster_id)
            # Intersect with original mask (not dilated)
            original_pixels = cluster_pixels & frame_mask
            if original_pixels.sum() >= min_cluster_size:
                clean_frame   |= original_pixels
                valid_clusters += 1

        noise_removed = frame_mask.sum() - clean_frame.sum()
        filtered_mask[:, t] = clean_frame.flatten()

        stats["clusters_per_frame"].append(valid_clusters)
        stats["removed_noise_px"].append(int(noise_removed))

    total_removed = sum(stats["removed_noise_px"])
    total_kept    = filtered_mask.sum()
    print(f"[spatial] MIN_CLUSTER_SIZE = {min_cluster_size} px")
    print(f"[spatial] Noise pixels removed: {total_removed}")
    print(f"[spatial] Anomalous kept      : {total_kept}")
    print(f"[spatial] Clusters per frame  : {stats['clusters_per_frame']}")

    return filtered_mask, stats
