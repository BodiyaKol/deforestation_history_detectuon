"""
anomaly_detection.py
────────────────────
Крок 3 — Виявлення аномалій з матриці S через Z-score.

Логіка:
  S = X − L містить все що відхилилось від стабільного фону.
  Але S включає і шум, і сезонні залишки, і справжні зміни.
  Z-score нормалізує кожен піксель по його власній історії змін —
  тому різкий одноразовий спад NDVI (вирубка) видно чітко,
  а повільні сезонні коливання відфільтровуються.

  Аномалія = Z < −2.0  (NDVI впав більш ніж на 2σ нижче норми пікселя)
           + піксель входить у forest_mask (це був ліс)
           + піксель НЕ входить у nonforest_mask (це не дорога/поле)

Вхід:  S, forest_mask, nonforest_mask, H, W
Вихід: S_masked, Z, anomaly_mask
"""

import numpy as np


# ── CONFIG ────────────────────────────────────────────────────────────────────
ANOMALY_Z_THRESHOLD = -2.5   # Z-score нижче якого вважаємо аномалією


# ─────────────────────────────────────────────────────────────────────────────
def compute_anomalies(
    S: np.ndarray,
    forest_mask: np.ndarray,
    nonforest_mask: np.ndarray,
    H: int,
    W: int,
    z_threshold: float = ANOMALY_Z_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Виявляє аномалії (вирубка, пожежа) у матриці залишків S.

    Алгоритм:
      1. Занулити пікселі nonforest_mask — не-ліс нас не цікавить
      2. Z-score по часовій осі: z[i,t] = (S[i,t] − mean_i) / std_i
         → кожен піксель нормалізується по своїй власній варіативності
      3. Аномалія: z[i,t] < threshold  AND  i ∈ forest_mask

    Parameters
    ----------
    S              : (pixels, frames) — матриця залишків X − L
    forest_mask    : (H, W) bool
    nonforest_mask : (H, W) bool
    H, W           : розміри знімку
    z_threshold    : поріг Z-score (від'ємний: падіння NDVI)

    Returns
    -------
    S_masked     : (pixels, frames) — S з нульовими не-лісовими пікселями
    Z            : (pixels, frames) — Z-score матриця
    anomaly_mask : (pixels, frames) bool — True = виявлена аномалія
    """
    # 1. Маскуємо не-ліс
    S_masked = S.copy()
    S_masked[nonforest_mask.flatten(), :] = 0.0

    # 2. Z-score по часовій осі (axis=1 → по кадрах для кожного пікселя)
    mean_s = S_masked.mean(axis=1, keepdims=True)
    std_s  = S_masked.std(axis=1,  keepdims=True) + 1e-8
    Z      = (S_masked - mean_s) / std_s

    # 3. Аномалія = різке падіння NDVI у лісових пікселях
    forest_flat  = forest_mask.flatten()[:, None]   # (pixels, 1) для broadcast
    anomaly_mask = (Z < z_threshold) & forest_flat

    n_events  = anomaly_mask.sum()
    n_pixels  = anomaly_mask.any(axis=1).sum()
    print(f"[anom] Z-threshold = {z_threshold}")
    print(f"[anom] Аномальних піксель-кадрів : {n_events}")
    print(f"[anom] Унікальних аномальних пікс: {n_pixels}")

    return S_masked, Z, anomaly_mask
