"""
Detect anomalies in residual matrix S using Z-score.
"""

import numpy as np
from scipy.ndimage import binary_dilation


# Config
ANOMALY_Z_THRESHOLD       = -1.0   # Z-score below which we consider anomaly
TRANSITION_Z_RELAX_FACTOR = 0.5    # for neighboring transition pixels
TRANSITION_DROP_THRESHOLD = -0.04  # total NDVI drop over period


def compute_anomalies(
    S: np.ndarray,
    forest_mask: np.ndarray,
    nonforest_mask: np.ndarray,
    H: int,
    W: int,
    z_threshold: float = ANOMALY_Z_THRESHOLD,
    X: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Detect anomalies in residual matrix S.

    Parameters
    S              : (pixels, frames) — residual matrix X - L
    forest_mask    : (H, W) bool
    nonforest_mask : (H, W) bool
    H, W           : image dimensions
    z_threshold    : Z-score threshold (negative: NDVI drop)

    Returns
    S_masked     : (pixels, frames) — S with zeroed non-forest pixels
    Z            : (pixels, frames) — Z-score matrix
    anomaly_mask : (pixels, frames) bool — True = detected anomaly
    """
    # Mask non-forest pixels
    S_masked = S.copy()
    S_masked[nonforest_mask.flatten(), :] = 0.0

    # Z-score along time axis
    mean_s = S_masked.mean(axis=1, keepdims=True)
    std_s  = S_masked.std(axis=1,  keepdims=True) + 1e-8
    Z      = (S_masked - mean_s) / std_s

    # Anomaly = sharp NDVI drop in forest pixels
    forest_flat  = forest_mask.flatten()[:, None]
    anomaly_mask = (Z < z_threshold) & forest_flat

    if X is not None:
        transition_mask = _build_transition_mask(
            anomaly_mask,
            X,
            forest_mask,
            H,
            W,
            Z,
            z_threshold=z_threshold,
        )
        anomaly_mask = anomaly_mask | transition_mask

    n_events  = anomaly_mask.sum()
    n_pixels  = anomaly_mask.any(axis=1).sum()
    print(f"[anom] Z-threshold = {z_threshold}")
    print(f"[anom] Anomalous pixel-frames: {n_events}")
    print(f"[anom] Unique anomalous pixels: {n_pixels}")

    return S_masked, Z, anomaly_mask


def _build_transition_mask(
    anomaly_mask: np.ndarray,
    X: np.ndarray,
    forest_mask: np.ndarray,
    H: int,
    W: int,
    Z: np.ndarray,
    z_threshold: float,
) -> np.ndarray:
    """
    Add transition pixels with moderate drop near sharp anomalies.
    """
    forest_flat = forest_mask.flatten()
    T = X.shape[1]
    transition_mask = np.zeros_like(anomaly_mask)
    structure = np.ones((3, 3), dtype=bool)

    total_drop = X[:, -1] - X[:, 0]
    decline_flat = (total_drop < TRANSITION_DROP_THRESHOLD) & forest_flat

    for t in range(T):
        frame_sharp = anomaly_mask[:, t].reshape(H, W)
        support = binary_dilation(frame_sharp, structure=structure)

        candidate = ((X[:, t] - X[:, 0]) < 0) & forest_flat
        relaxed = (Z[:, t] < (z_threshold * TRANSITION_Z_RELAX_FACTOR)) & forest_flat[:, None]
        merged = support.flatten() & candidate & relaxed[:, 0]

        transition_mask[:, t] = merged | (decline_flat & support.flatten())

    # Make persistent transitions from first occurrence
    persistent = transition_mask.any(axis=1)
    persistent_indices = np.where(persistent)[0]
    for i in persistent_indices:
        first_t = np.argmax(transition_mask[i, :])
        transition_mask[i, first_t:] = True

    n_transitions = transition_mask.sum()
    if n_transitions > 0:
        print(f"[anom] Transition pixels added: {n_transitions}")

    return transition_mask
