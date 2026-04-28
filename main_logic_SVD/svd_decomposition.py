"""
Custom truncated SVD implementation using power iteration method.
"""

import numpy as np

SVD_VARIANCE_THRESHOLD = 0.995
POWER_ITER             = 30
MAX_COMPONENTS         = 10
CONVERGENCE_TOL        = 1e-12
BASELINE_WINDOW = 3
VARIANCE_THRESHOLD = 0.95
MIN_STD = 0.04

def _power_iteration(A: np.ndarray, n_iter: int = POWER_ITER, seed: int = 0) -> tuple:
    """
    Finds the FIRST singular triplet (u, sigma, v) of matrix A
    using power iterations.

    Idea:
      v_(t+1) = A^tA · v_t / ||A^tA · v_t||
      Two-step variant to avoid computing A^tA explicitly:
        q = A·v,  v_new = A^t·q,  v = v_new/||v_new||
    """
    rng = np.random.default_rng(seed)
    n   = A.shape[1]
    v   = rng.standard_normal(n)
    v  /= np.linalg.norm(v) + 1e-14

    for _ in range(n_iter):
        q     = A @ v
        v_new = A.T @ q
        nrm   = np.linalg.norm(v_new)
        if nrm < 1e-14:
            break
        v_new /= nrm
        if np.linalg.norm(v_new - v) < CONVERGENCE_TOL:
            v = v_new; break
        v = v_new

    Av    = A @ v
    sigma = np.linalg.norm(Av)
    u     = Av / sigma if sigma > 1e-14 else rng.standard_normal(A.shape[0])
    return u, float(sigma), v


def _deflate(A, u, sigma, v):
    """A_new = A − sigma · u · v^t  (rank-1 deflation)"""
    return A - sigma * np.outer(u, v)


def _choose_rank(sigmas: np.ndarray, threshold: float) -> int:
    s2    = sigmas ** 2
    total = s2.sum()
    if total < 1e-14:
        return 1
    cum = np.cumsum(s2) / total
    k   = int(np.searchsorted(cum, threshold)) + 1
    return max(1, min(k, len(sigmas)))


def _prepare_svd_matrix(
    X: np.ndarray,
    forest_mask: np.ndarray | None,
    nonforest_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Prepares input matrix for SVD to focus model on forest.

    - Makes nonforest pixels constant over time
    - Centers each pixel by first frame
    - Amplifies long-term drops for forest pixels
    - Reduces seasonal/positive deviations so SVD doesn't average changes
    """
    if forest_mask is None or nonforest_mask is None:
        baseline = np.zeros((X.shape[0], 1), dtype=np.float64)
        return X.astype(np.float64), baseline

    X_init = X.astype(np.float64).copy()
    nonforest_flat = nonforest_mask.flatten()
    X_init[nonforest_flat, :] = X_init[nonforest_flat, 0][:, None]
    baseline = X_init[:, 0:1]

    X_centered = X_init - baseline
    forest_flat = forest_mask.flatten()

    # Amplify long-term negative trends in forest pixels,
    # so SVD pays more attention to real deforestation.
    forest_vals = X_centered[forest_flat, :]
    long_drop = forest_vals[:, -1] < -0.02
    if np.any(long_drop):
        forest_vals[long_drop, :] *= 1.25

    # Reduce positive fluctuations in forest histograms,
    # so seasonal rises don't become part of background L.
    forest_vals = np.where(forest_vals > 0.0, forest_vals * 0.5, forest_vals)
    X_centered[forest_flat, :] = forest_vals

    return X_centered, baseline


def compute_svd_background(
    X: np.ndarray,
    forest_mask: np.ndarray,
    nonforest_mask: np.ndarray,
    window: int = BASELINE_WINDOW,
    variance_threshold: float = VARIANCE_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Computes baseline via MANUAL truncated SVD on first frames
    using Power Iteration + Deflation.

    Returns:
      L_full        : (pixels, frames) full baseline model for entire series
      baseline_std  : (pixels,) std of residuals on baseline
      sigma         : singular values of chosen rank
      k             : chosen rank
    """
    # беремо лише перші кадри як baseline-вікно
    X_baseline = X[:, :window]

    # та сама підготовка даних
    X_for_svd, baseline = _prepare_svd_matrix(
        X_baseline,
        forest_mask,
        nonforest_mask
    )

    # копія матриці для дефляції
    A = X_for_svd.copy()

    max_rank = min(MAX_COMPONENTS, min(A.shape))
    Us = []
    sigmas = []

    total_variance = float(np.sum(A ** 2))
    accumulated_var = 0.0

    # -----------------------------
    # Ручний SVD через Power Iteration
    # -----------------------------
    for idx in range(max_rank):
        u, s, v = _power_iteration(A, n_iter=POWER_ITER, seed=idx)

        if s < 1e-12:
            break

        Us.append(u)
        sigmas.append(s)

        accumulated_var += s ** 2
        explained = accumulated_var / (total_variance + 1e-14)

        # вибір рангу по variance threshold
        if explained >= variance_threshold:
            break

        A = _deflate(A, u, s, v)

    # якщо нічого не знайшли
    if len(sigmas) == 0:
        sigmas = np.array([0.0])
        U_k = np.zeros((X.shape[0], 1))
        k = 1
    else:
        sigmas = np.array(sigmas, dtype=np.float64)
        U_k = np.column_stack(Us)
        k = U_k.shape[1]

    # -----------------------------
    # Проєкція всієї часової серії
    # -----------------------------
    X_centered = X.astype(np.float64) - baseline

    coefficients = U_k.T @ X_centered
    L_full = U_k @ coefficients + baseline

    # -----------------------------
    # Шум baseline-вікна
    # -----------------------------
    residual_baseline = X_baseline - L_full[:, :window]
    baseline_std = np.std(residual_baseline, axis=1)
    baseline_std = np.maximum(baseline_std, MIN_STD)

    return L_full, baseline_std, sigmas, k
