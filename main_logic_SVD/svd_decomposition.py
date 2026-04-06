"""
svd_decomposition.py
────────────────────
Власна реалізація усіченого SVD через степеневий метод (Power Iteration).
"""

import numpy as np

SVD_VARIANCE_THRESHOLD = 0.995
POWER_ITER             = 30
MAX_COMPONENTS         = 10
CONVERGENCE_TOL        = 1e-12


def _power_iteration(A: np.ndarray, n_iter: int = POWER_ITER, seed: int = 0) -> tuple:
    """
    Знаходить ПЕРШУ сингулярну трійку (u, σ, v) матриці A
    методом степеневих ітерацій.

    Ідея:
      v_(t+1) = AᵀA · v_t / ||AᵀA · v_t||
      Двокроковий варіант щоб не рахувати AᵀA явно:
        q = A·v,  v_new = Aᵀ·q,  v = v_new/||v_new||
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
    """A_new = A − σ · u · vᵀ  (rank-1 deflation)"""
    return A - sigma * np.outer(u, v)


def _choose_rank(sigmas: np.ndarray, threshold: float) -> int:
    s2    = sigmas ** 2
    total = s2.sum()
    if total < 1e-14:
        return 1
    cum = np.cumsum(s2) / total
    k   = int(np.searchsorted(cum, threshold)) + 1
    return max(1, min(k, len(sigmas)))


def compute_svd_background(
    X: np.ndarray,
    variance_threshold: float = SVD_VARIANCE_THRESHOLD,
    power_iter: int            = POWER_ITER,
    max_components: int        = MAX_COMPONENTS,
) -> tuple:
    """
    Будує матрицю фону L через усічений SVD (Power Iteration + Deflation).

    L = Σᵢ₌₁ᵏ  σᵢ · uᵢ · vᵢᵀ    (стабільний фон)
    S = X − L                      (аномалії)

    Ключовий момент:
      - Знаходимо компоненти одну за одною
      - Кожного разу робимо deflation: A ← A − σᵢuᵢvᵢᵀ
      - Зупиняємось коли накопичена дисперсія >= variance_threshold
      - MIN 2 компоненти — перша описує загальний рівень NDVI,
        друга — сезонну варіацію. Без мінімум 2 компонент
        вирубка може потрапити у першу компоненту.
    """
    print(f"[svd]  Power Iteration SVD (max_k={max_components}, iters={power_iter}) ...")

    # Важливо: рахуємо total_variance від ОРИГІНАЛЬНОГО X (до deflation)
    total_variance = float(np.sum(X ** 2))

    A      = X.astype(np.float64).copy()
    Us, Vs, sigmas = [], [], []
    accumulated_var = 0.0

    for idx in range(max_components):
        u, s, v = _power_iteration(A, n_iter=power_iter, seed=idx)
        Us.append(u); Vs.append(v); sigmas.append(s)
        accumulated_var += s ** 2

        explained = accumulated_var / (total_variance + 1e-14)

        # Мінімум 2 компоненти, потім зупиняємось по дисперсії
        if idx >= 1 and explained >= variance_threshold:
            break

        A = _deflate(A, u, s, v)

    all_sigmas = np.array(sigmas)
    k          = len(sigmas)   # вже обраний ранг
    explained  = accumulated_var / (total_variance + 1e-14)

    print(f"[svd]  Ранг k = {k}  |  пояснена дисперсія = {min(explained, 1.0):.4f}")
    print(f"[svd]  Перші {min(5, k)} σ: {all_sigmas[:5].round(3)}")

    # Реконструкція L
    L = np.zeros_like(X, dtype=np.float64)
    for i in range(k):
        L += sigmas[i] * np.outer(Us[i], Vs[i])

    S = X - L
    return L, S, all_sigmas, k
