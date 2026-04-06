"""
forest_masks.py
───────────────
Крок 2 — Побудова просторових масок лісу на основі першого знімку.

Навіщо маски:
  SVD "вивчає" патерни з усіх знімків. Якщо на перших знімках вже є
  вирубка — алгоритм сприйме її як нормальний фон і пропустить.
  Маски дозволяють:
    1. Слідкувати ТІЛЬКИ за пікселями що спочатку були лісом
    2. Прибрати поля / дороги / будівлі з аномалій (вони нас не цікавлять)

Вхід:  X, H, W
Вихід: forest_mask, nonforest_mask  — булеві масиви (H × W)
"""

import numpy as np


# ── CONFIG ────────────────────────────────────────────────────────────────────
NDVI_FOREST_THRESHOLD    = 0.4   # NDVI > цього → вважається лісом
NDVI_NONFOREST_THRESHOLD = 0.2   # NDVI < цього → точно не ліс


# ─────────────────────────────────────────────────────────────────────────────
def build_forest_masks(
    X: np.ndarray,
    H: int,
    W: int,
    forest_threshold: float    = NDVI_FOREST_THRESHOLD,
    nonforest_threshold: float = NDVI_NONFOREST_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Будує маски лісу та не-лісу за ПЕРШИМ знімком.

    Використовуємо перший знімок як "базовий стан":
      - forest_mask    : де NDVI > 0.4  → початковий ліс
      - nonforest_mask : де NDVI < 0.2  → завжди не ліс

    Parameters
    ----------
    X                   : (pixels, frames) — NDVI матриця
    H, W                : висота та ширина одного знімку
    forest_threshold    : поріг NDVI для лісу
    nonforest_threshold : поріг NDVI для не-лісу

    Returns
    -------
    forest_mask    : (H, W) bool — пікселі початкового лісу
    nonforest_mask : (H, W) bool — пікселі що ніколи не були лісом
    """
    first_frame = X[:, 0].reshape(H, W)

    forest_mask    = first_frame > forest_threshold
    nonforest_mask = first_frame < nonforest_threshold

    total          = H * W
    forest_px      = forest_mask.sum()
    nonforest_px   = nonforest_mask.sum()

    print(f"[mask] Ліс на першому знімку   : {forest_px} px  ({100 * forest_px / total:.1f}%)")
    print(f"[mask] Не-ліс на першому знімку: {nonforest_px} px  ({100 * nonforest_px / total:.1f}%)")

    return forest_mask, nonforest_mask
