"""
diagnostics.py
──────────────
Діагностичні графіки для перевірки якості SVD розкладу.

Зберігає diagnostics.png у output/ — дивись його після кожного запуску
щоб переконатись що L і S виглядають розумно.

Три панелі:
  1. Сингулярні значення σ — "ліктьова" точка показує де обрізати ранг
  2. Залишки S на середньому кадрі — мають бути нулі скрізь крім аномалій
  3. Сумарна карта аномалій — де хоч раз за весь період виявлено зміну
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
def plot_diagnostics(
    sigma: np.ndarray,
    X: np.ndarray,
    L: np.ndarray,
    Z: np.ndarray,
    anomaly_mask: np.ndarray,
    H: int,
    W: int,
    dates: list,
    k: int,
    output_dir: Path,
) -> None:
    """
    Будує та зберігає діагностичний графік (3 панелі).

    Parameters
    ----------
    sigma        : всі сингулярні значення
    X            : оригінальна матриця (pixels, frames)
    L            : low-rank фон
    Z            : Z-score матриця
    anomaly_mask : бінарна маска аномалій
    H, W         : розміри знімку
    dates        : список дат
    k            : обраний ранг SVD
    output_dir   : куди зберігати
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("SVD Deforestation — Diagnostics", fontsize=13, fontweight="bold")

    # ── 1. Сингулярні значення ──────────────────────────────────────────────
    ax = axes[0]
    n_show = min(30, len(sigma))
    ax.semilogy(range(n_show), sigma[:n_show], "o-", color="#2d6a4f", linewidth=1.5, markersize=4)
    ax.axvline(x=k - 1, color="#e63946", linestyle="--", linewidth=1.5, label=f"k = {k}")
    ax.set_title("Сингулярні значення σ")
    ax.set_xlabel("Компонента")
    ax.set_ylabel("σ  (log scale)")
    ax.legend()
    ax.grid(True, alpha=0.25)

    # ── 2. Залишки S на середньому кадрі ────────────────────────────────────
    mid = len(dates) // 2
    residuals = (X[:, mid] - L[:, mid]).reshape(H, W)
    ax = axes[1]
    im = ax.imshow(residuals, cmap="RdYlGn", vmin=-0.3, vmax=0.3)
    ax.set_title(f"S = X − L  (кадр {mid} / {dates[mid]})")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # ── 3. Сумарна карта аномалій ────────────────────────────────────────────
    total_anomaly = anomaly_mask.any(axis=1).reshape(H, W)
    ax = axes[2]
    im2 = ax.imshow(total_anomaly.astype(float), cmap="Reds", vmin=0, vmax=1)
    ax.set_title(f"Сумарна карта вирубки ({dates[0]} → {dates[-1]})")
    ax.axis("off")
    plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    out_path = output_dir / "diagnostics.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[diag] Збережено: {out_path}")

