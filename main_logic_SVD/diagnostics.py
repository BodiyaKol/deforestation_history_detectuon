"""
Diagnostic plots for checking SVD decomposition quality.

Saves diagnostics.png in output/ — check it after each run
to ensure L and S look reasonable.

Three panels:
  1. Singular values σ — "elbow" point shows where to truncate rank
  2. Residuals S on middle frame — should be zero everywhere except anomalies
  3. Total anomaly map — where change was detected at least once
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
    title_suffix: str = "SVD",
) -> None:
    """
    Builds and saves diagnostic plot (3 panels).

    Parameters
    ----------
    sigma        : all singular values (may be empty for Change Detection)
    X            : original matrix (pixels, frames)
    L            : low-rank background
    Z            : Z-score matrix
    anomaly_mask : binary anomaly mask
    H, W         : image dimensions
    dates        : list of dates
    k            : chosen SVD rank (0 for Change Detection)
    output_dir   : where to save
    title_suffix : by which method computed ("SVD" or "Change Detection")
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Deforestation Diagnostics [{title_suffix}]", fontsize=13, fontweight="bold")

    # 1. Singular values (or Z-score stats for Change Detection)
    ax = axes[0]
    if len(sigma) > 0:
        # SVD method
        n_show = min(30, len(sigma))
        ax.semilogy(range(n_show), sigma[:n_show], "o-", color="#2d6a4f", linewidth=1.5, markersize=4)
        ax.axvline(x=k - 1, color="#e63946", linestyle="--", linewidth=1.5, label=f"k = {k}")
        ax.set_title("Singular values σ")
        ax.set_ylabel("σ  (log scale)")
    else:
        # Change Detection method
        z_values = Z[~np.isnan(Z) & ~np.isinf(Z)].flatten()
        ax.hist(z_values, bins=50, color="#2d6a4f", alpha=0.7, edgecolor="black")
        ax.axvline(x=-2.5, color="#e63946", linestyle="--", linewidth=1.5, label="Threshold")
        ax.set_title("Z-score distribution")
        ax.set_ylabel("Frequency")
    ax.set_xlabel("Component" if len(sigma) > 0 else "Z-score")
    ax.legend()
    ax.grid(True, alpha=0.25)

    # 2. Residuals on middle frame
    mid = len(dates) // 2
    residuals = (X[:, mid] - L[:, mid]).reshape(H, W)
    ax = axes[1]
    im = ax.imshow(residuals, cmap="RdYlGn", vmin=-0.3, vmax=0.3)
    ax.set_title(f"S = X − L  (frame {mid} / {dates[mid]})")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 3. Total anomaly map
    total_anomaly = anomaly_mask.any(axis=1).reshape(H, W)
    ax = axes[2]
    im2 = ax.imshow(total_anomaly.astype(float), cmap="Reds", vmin=0, vmax=1)
    ax.set_title(f"Total deforestation map ({dates[0]} → {dates[-1]})")
    ax.axis("off")
    plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    out_path = output_dir / "diagnostics.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[diag] Saved: {out_path}")

