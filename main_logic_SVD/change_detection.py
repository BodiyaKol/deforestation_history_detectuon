"""
SVD + regression change detection.

Uses SVD for baseline from first frames, then detects gradual NDVI declines via regression.
"""

import numpy as np
from main_logic_SVD.svd_decomposition import _prepare_svd_matrix, _choose_rank, compute_svd_background

BASELINE_WINDOW = 3
VARIANCE_THRESHOLD = 0.95
SLOPE_THRESHOLD = -0.01
DROP_THRESHOLD = -0.05
Z_THRESHOLD = -1.7
PERSISTENCE_FRAMES = 3
MIN_STD = 0.04
MEAN_LATE_THRESHOLD = -0.5


def compute_baseline(
    X: np.ndarray,
    forest_mask: np.ndarray,
    nonforest_mask: np.ndarray,
    window: int = BASELINE_WINDOW,
    variance_threshold: float = VARIANCE_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Computes baseline via SVD on first frames.

    Returns:
      L_full        : (pixels, frames) full baseline model for entire series
      baseline_std  : (pixels,) std of residuals on baseline
      sigma         : singular values of SVD baseline
      k             : chosen rank
    """
    X_baseline = X[:, :window]
    X_for_svd, baseline = _prepare_svd_matrix(X_baseline, forest_mask, nonforest_mask)
    U, s, Vt = np.linalg.svd(X_for_svd, full_matrices=False)
    k = _choose_rank(s, variance_threshold)
    U_k = U[:, :k]

    X_centered = X.astype(np.float64) - baseline
    coefficients = U_k.T @ X_centered
    L_full = U_k @ coefficients + baseline

    residual_baseline = X_baseline - L_full[:, :window]
    baseline_std = np.std(residual_baseline, axis=1)
    baseline_std = np.maximum(baseline_std, MIN_STD)
    return L_full, baseline_std, s[:k], k


def solve_least_squares(A: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    FULL manual least squares solver.

    Solves:
        min ||A·coef - Y||

    A : shape (T, 2)
    Y : shape (pixels, T)

    Returns:
        coef : shape (2, pixels)

    No np.linalg, no np.dot, no QR from numpy.
    """

    A = A.astype(float)
    Y = Y.astype(float)

    rows = A.shape[0]      # T
    cols = A.shape[1]      # 2
    pixels = Y.shape[0]

    # -----------------------------------
    # Helper functions
    # -----------------------------------
    def manual_dot(v1, v2):
        s = 0.0
        for i in range(len(v1)):
            s += v1[i] * v2[i]
        return s

    def manual_norm(v):
        s = 0.0
        for i in range(len(v)):
            s += v[i] * v[i]
        return s ** 0.5

    def get_col(M, j):
        col = np.zeros(M.shape[0], dtype=float)
        for i in range(M.shape[0]):
            col[i] = M[i, j]
        return col

    def set_col(M, j, v):
        for i in range(len(v)):
            M[i, j] = v[i]

    # -----------------------------------
    # Classical Gram-Schmidt QR
    # -----------------------------------
    Q = np.zeros((rows, cols), dtype=float)
    R = np.zeros((cols, cols), dtype=float)

    for j in range(cols):
        v = get_col(A, j)

        for i in range(j):
            qi = get_col(Q, i)
            aj = get_col(A, j)

            proj = manual_dot(qi, aj)
            R[i, j] = proj

            for k in range(rows):
                v[k] -= proj * qi[k]

        norm_v = manual_norm(v)

        if norm_v < 1e-12:
            raise ValueError("Columns are linearly dependent.")

        R[j, j] = norm_v

        for k in range(rows):
            v[k] /= norm_v

        set_col(Q, j, v)

    # -----------------------------------
    # Compute B = Qᵀ · Yᵀ
    # shape = (2, pixels)
    # -----------------------------------
    B = np.zeros((cols, pixels), dtype=float)

    for i in range(cols):
        qi = get_col(Q, i)

        for p in range(pixels):
            s = 0.0
            for t in range(rows):
                s += qi[t] * Y[p, t]
            B[i, p] = s

    # -----------------------------------
    # Solve R·coef = B
    # Manual back substitution
    # -----------------------------------
    coef = np.zeros((cols, pixels), dtype=float)

    for p in range(pixels):
        for i in range(cols - 1, -1, -1):
            s = B[i, p]

            for j in range(i + 1, cols):
                s -= R[i, j] * coef[j, p]

            coef[i, p] = s / R[i, i]

    return coef


def fit_slopes(R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Solve linear regression for each pixel in matrix form."""
    T = R.shape[1]
    t = np.arange(T, dtype=np.float64)
    A = np.vstack([np.ones(T, dtype=np.float64), t]).T
    coef = solve_least_squares(A, R)
    intercept = coef[0, :]
    slope = coef[1, :]
    return slope, intercept


def compute_regression_changes(
    X: np.ndarray,
    forest_mask: np.ndarray,
    nonforest_mask: np.ndarray,
    H: int,
    W: int,
    baseline_window: int = BASELINE_WINDOW,
    slope_threshold: float = SLOPE_THRESHOLD,
    drop_threshold: float = DROP_THRESHOLD,
    z_threshold: float = Z_THRESHOLD,
    persistence_frames: int = PERSISTENCE_FRAMES,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Detects gradual NDVI changes.

    1. Builds baseline from first frames using SVD.
    2. Computes residuals from baseline.
    3. Performs regression for each pixel.
    4. Marks long negative trends confirmed by consecutive drops.
    """
    if verbose:
        print(f"\n[regression] SVD baseline + regression detection")
        print(f"[regression] baseline window = {baseline_window} frames")

    L_full, baseline_std, sigma, k = compute_svd_background(
        X, forest_mask, nonforest_mask, window=baseline_window
    )
    T = X.shape[1]
    pixels = X.shape[0]

    R = X - L_full
    Z = R / baseline_std[:, None]

    nonforest_flat = nonforest_mask.flatten()
    Z[nonforest_flat, :] = 0.0

    forest_flat = forest_mask.flatten()
    if T - baseline_window >= 2:
        slopes, _ = fit_slopes(R[:, baseline_window:])
    else:
        slopes, _ = fit_slopes(R)

    total_drop = R[:, -1]
    mean_late = np.mean(Z[:, baseline_window:], axis=1)
    valid = (
        forest_flat
        & (slopes < slope_threshold)
        & (total_drop < drop_threshold)
        & (mean_late < MEAN_LATE_THRESHOLD)
    )

    anomaly_mask = np.zeros_like(Z, dtype=bool)
    change_time = np.full(pixels, -1, dtype=int)
    sharp = Z < z_threshold

    valid_idx = np.nonzero(valid)[0]
    if valid_idx.size > 0:
        for t in range(baseline_window, T - persistence_frames + 1):
            window_hits = np.all(sharp[valid_idx, t:t + persistence_frames], axis=1)
            if not np.any(window_hits):
                continue
            hit_pixels = valid_idx[window_hits]
            anomaly_mask[hit_pixels, t:] = True
            new_hits = hit_pixels[change_time[hit_pixels] == -1]
            change_time[new_hits] = t

    n_events = int(anomaly_mask.sum())
    n_pixels = int(anomaly_mask.any(axis=1).sum())
    detected_changes = int((change_time != -1).sum())

    if verbose:
        print(f"[regression] negative slopes       : {valid.sum()}")
        print(f"[regression] detected pixels       : {detected_changes}")
        print(f"[regression] anomaly pixel-frames  : {n_events}")
        print(f"[regression] unique anomaly pixels : {n_pixels}")

    return anomaly_mask, Z, change_time
