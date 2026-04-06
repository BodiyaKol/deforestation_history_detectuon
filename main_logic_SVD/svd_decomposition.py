"""
Custom truncated SVD implementation using power iteration method.
"""

import numpy as np

SVD_VARIANCE_THRESHOLD = 0.995
POWER_ITER             = 30
MAX_COMPONENTS         = 10
CONVERGENCE_TOL        = 1e-12


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
    variance_threshold: float = SVD_VARIANCE_THRESHOLD,
    power_iter: int            = POWER_ITER,
    max_components: int        = MAX_COMPONENTS,
    forest_mask: np.ndarray | None = None,
    nonforest_mask: np.ndarray | None = None,
) -> tuple:
    """
    Builds background matrix L via truncated SVD (Power Iteration + Deflation).
    """
    effective_threshold = variance_threshold
    effective_max_components = max_components
    if forest_mask is not None and nonforest_mask is not None:
        effective_threshold = min(variance_threshold, 0.99)
        effective_max_components = min(max_components, 6)

    print(f"[svd]  Power Iteration SVD (max_k={effective_max_components}, iters={power_iter}) ...")

    X_for_svd, baseline = _prepare_svd_matrix(X, forest_mask, nonforest_mask)
    total_variance = float(np.sum(X_for_svd ** 2))

    A      = X_for_svd.copy()
    Us, Vs, sigmas = [], [], []
    accumulated_var = 0.0

    for idx in range(effective_max_components):
        u, s, v = _power_iteration(A, n_iter=power_iter, seed=idx)
        Us.append(u); Vs.append(v); sigmas.append(s)
        accumulated_var += s ** 2

        explained = accumulated_var / (total_variance + 1e-14)

        # Minimum 2 components, then stop by variance
        if idx >= 1 and explained >= effective_threshold:
            break

        A = _deflate(A, u, s, v)

    all_sigmas = np.array(sigmas)
    k          = len(sigmas)   # already chosen rank
    explained  = accumulated_var / (total_variance + 1e-14)

    print(f"[svd]  Rank k = {k}  |  explained variance = {min(explained, 1.0):.4f}")
    print(f"[svd]  First {min(5, k)} σ: {all_sigmas[:5].round(3)}")

    # Reconstruct L in centered space
    L = np.zeros_like(X_for_svd, dtype=np.float64)
    for i in range(k):
        L += sigmas[i] * np.outer(Us[i], Vs[i])

    # Return L to original scale
    L += baseline
    S = X - L
    return L, S, all_sigmas, k
