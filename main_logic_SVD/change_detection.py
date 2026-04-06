"""
change_detection.py
───────────────────
SVD + regression change detection.

Ми використовуємо SVD лише для побудови baseline з перших кадрів,
а потім шукаємо плавні зниження NDVI в кожному пікселі через регресію.
"""

import numpy as np
from main_logic_SVD.svd_decomposition import _prepare_svd_matrix, _choose_rank

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
    Обчислює baseline через SVD на перших кадрах.

    Returns:
      L_full        : (pixels, frames) повна baseline-модель для всього ряду
      baseline_std  : (pixels,) стандартне відхилення залишків на baseline
      sigma         : сингулярні значення SVD baseline
      k             : обраний ранг
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
    """Розв'язуємо min ||A·coef - Y|| у least squares для всіх стовпців Y."""
    # A: (T, 2), Y: (pixels, T)
    Q, R = np.linalg.qr(A)
    coef = np.linalg.solve(R, Q.T @ Y.T)
    return coef


def fit_slopes(R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Розв'язуємо лінійну регресію для кожного пікселя у матричній формі."""
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
    Детекція плавних змін NDVI.

    1. Створюємо baseline з перших кадрів за допомогою SVD.
    2. Комп'ютуємо залишки від baseline.
    3. Виконуємо регресію для кожного пікселя.
    4. Помічаємо довгі негативні тренди, підтверджені послідовними падіннями.
    """
    if verbose:
        print(f"\n[regression] SVD baseline + regression detection")
        print(f"[regression] baseline window = {baseline_window} кадрів")

    L_full, baseline_std, sigma, k = compute_baseline(
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
