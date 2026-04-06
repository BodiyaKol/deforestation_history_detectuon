"""
io_handler.py
─────────────
Завантаження даних від Максима та збереження результатів для Юлії.

Вхідні файли (data/):
  X.npy      — матриця (pixels × frames)
  meta.npy   — словник: dates, height, width, bbox

Вихідні файли (output/):
  L.npy            — low-rank фон
  S.npy            — аномалії (сирі залишки, маскований не-ліс)
  Z.npy            — Z-score матриця
  anomaly_mask.npy — бінарна маска вирубки/пожеж
  output_meta.npy  — метадані для відео (дати, розміри, пороги)
"""

import numpy as np
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
def load_input(data_dir: Path) -> tuple[np.ndarray, int, int, list]:
    """
    Завантажує X та метадані від Максима.

    Returns
    -------
    X     : (pixels, frames)
    H, W  : розміри одного знімку
    dates : список рядків формату 'YYYY-MM-DD'
    """
    X_path    = data_dir / "X.npy"
    meta_path = data_dir / "meta.npy"

    if not X_path.exists():
        raise FileNotFoundError(f"Не знайдено {X_path} — переконайся що Максим поклав файли у data/")
    if not meta_path.exists():
        raise FileNotFoundError(f"Не знайдено {meta_path}")

    X    = np.load(X_path)
    meta = np.load(meta_path, allow_pickle=True).item()

    H     = meta["height"]
    W     = meta["width"]
    dates = meta["dates"]

    print(f"[io]   X завантажено: {X.shape}  ({H}×{W} px, {len(dates)} кадрів)")
    print(f"[io]   Період: {dates[0]}  →  {dates[-1]}")

    return X, H, W, dates


# ─────────────────────────────────────────────────────────────────────────────
def save_outputs(
    output_dir: Path,
    L: np.ndarray,
    S: np.ndarray,
    Z: np.ndarray,
    anomaly_mask: np.ndarray,
    dates: list,
    H: int,
    W: int,
    z_threshold: float,
    ndvi_forest_threshold: float,
) -> None:
    """
    Зберігає всі матриці для Юлії.

    Структура output/:
      L.npy            ← стабільний фон (для overlay у відео)
      S.npy            ← сирі залишки
      Z.npy            ← нормалізована інтенсивність аномалій
      anomaly_mask.npy ← True/False маска вирубки
      output_meta.npy  ← всі метадані в одному словнику
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "L.npy",            L)
    np.save(output_dir / "S.npy",            S)
    np.save(output_dir / "Z.npy",            Z)
    np.save(output_dir / "anomaly_mask.npy", anomaly_mask)
    np.save(output_dir / "output_meta.npy",  {
        "dates":                 dates,
        "height":                H,
        "width":                 W,
        "z_threshold":           z_threshold,
        "ndvi_forest_threshold": ndvi_forest_threshold,
    })

    print(f"[io]   Збережено у {output_dir}/")
    print(f"[io]     L.npy            {L.shape}")
    print(f"[io]     S.npy            {S.shape}")
    print(f"[io]     Z.npy            {Z.shape}")
    print(f"[io]     anomaly_mask.npy {anomaly_mask.shape}")
    print(f"[io]     output_meta.npy")
    