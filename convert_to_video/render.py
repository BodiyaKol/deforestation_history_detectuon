"""
convert_to_video/render.py
──────────────────────────
Visualization and video output.

Reads output/ (results of the SVD pipeline) and generates:
  1. deforestation.gif           — frame-by-frame Z-score animation
  2. deforestation_overlay.gif   — animation with anomaly mask overlay
  3. total_deforestation_map.png — cumulative deforestation map
  4. ndvi_change_map.png         — NDVI change map (first → last frame)

Runs automatically from main.py (Step 3).
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import imageio.v2 as imageio


# ── Шляхи ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("output")
VIDEO_DIR  = Path("output") / "video"


# ─────────────────────────────────────────────────────────────────────────────
def load_data():
    Z            = np.load(OUTPUT_DIR / "Z.npy")
    L            = np.load(OUTPUT_DIR / "L.npy")
    anomaly_mask = np.load(OUTPUT_DIR / "anomaly_mask.npy")
    meta         = np.load(OUTPUT_DIR / "output_meta.npy", allow_pickle=True).item()

    H     = meta["height"]
    W     = meta["width"]
    dates = meta["dates"]

    print(f"[video] Loaded: Z{Z.shape}, anomaly{anomaly_mask.shape}")
    print(f"[video] Frames: {len(dates)}, size: {H}×{W}")
    return Z, L, anomaly_mask, H, W, dates


# ─────────────────────────────────────────────────────────────────────────────
def make_zscore_frames(Z, H, W, dates):
    """
    Frames with a Z-score heatmap.
    Green = normal, red = anomaly (deforestation).
    """
    frames = []
    for i in range(Z.shape[1]):
        frame = Z[:, i].reshape(H, W)

        fig, ax = plt.subplots(figsize=(9, 7), facecolor="#0f1117")
        ax.set_facecolor("#0f1117")

        im = ax.imshow(frame, cmap="RdYlGn", vmin=-2.0, vmax=2.0,
                       interpolation="bilinear")

        cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cbar.set_label("Z-score", color="white", fontsize=10)
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

        ax.set_title(
            f"Forest change — {dates[i]}",
            color="white", fontsize=14, fontweight="bold", pad=12
        )
        ax.set_xlabel("← West    East →", color="#aaaaaa", fontsize=9)
        ax.set_ylabel("North ↑", color="#aaaaaa", fontsize=9)
        ax.tick_params(colors="#555555")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")

        # Підпис кадру
        ax.text(0.02, 0.02, f"Frame {i+1}/{Z.shape[1]}",
                transform=ax.transAxes, color="#c7c7c7",
                fontsize=9, va="bottom",
                bbox=dict(facecolor="#0f1117", edgecolor="#333333", alpha=0.75, pad=2))

        plt.tight_layout()
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        img = np.asarray(buf)[:, :, :3]
        frames.append(img.copy())
        plt.close()

    return frames


# ─────────────────────────────────────────────────────────────────────────────
def make_overlay_frames(Z, anomaly_mask, H, W, dates):
    """
    Frames where an anomaly mask is overlaid on the NDVI background (red).
    More visually clear for presentation.
    """
    frames = []

    # Нормалізуємо Z для фону (0..1)
    z_norm = np.clip((Z + 3) / 6, 0, 1)

    for i in range(Z.shape[1]):
        bg     = z_norm[:, i].reshape(H, W)
        mask2d = anomaly_mask[:, i].reshape(H, W)

        # RGB background — stronger green palette
        cmap = plt.cm.Greens
        bg_rgb = (cmap(bg)[:, :, :3] * 255).astype(np.uint8)

        # Red overlay for anomalies
        overlay = bg_rgb.copy()
        overlay[mask2d] = [255, 50, 50]   # bright red

        fig, ax = plt.subplots(figsize=(9, 7), facecolor="#0f1117")
        ax.set_facecolor("#0f1117")
        ax.imshow(overlay, interpolation="bilinear")

        ax.set_title(
            f"Detected forest changes — {dates[i]}",
            color="white", fontsize=14, fontweight="bold", pad=12
        )

        # Легенда
        legend_els = [
            Patch(facecolor="#22c55e", label="Healthy forest"),
            Patch(facecolor="#dc2626", label="Anomaly (deforestation/fire)"),
        ]
        ax.legend(handles=legend_els, loc="lower right",
                  facecolor="#1e2130", edgecolor="#444", labelcolor="white",
                  fontsize=10)

        n_anom = mask2d.sum()
        ax.text(0.02, 0.98,
                f"Anomalous pixels: {n_anom:,}",
                transform=ax.transAxes, color="#ffd166",
                fontsize=10, va="top", fontweight="bold",
                bbox=dict(facecolor="#0f1117", edgecolor="#444", alpha=0.85, pad=3))

        ax.axis("off")
        plt.tight_layout()
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        img = np.asarray(buf)[:, :, :3]
        frames.append(img.copy())
        plt.close()

    return frames


# ─────────────────────────────────────────────────────────────────────────────
def make_total_map(anomaly_mask, Z, H, W, dates):
    """
    Static map — places where an anomaly appeared at least once.
    + intensity (how many frames were anomalous).
    """
    total_anomaly = anomaly_mask.any(axis=1).reshape(H, W)
    intensity     = anomaly_mask.sum(axis=1).reshape(H, W).astype(float)
    intensity_norm = intensity / max(anomaly_mask.shape[1], 1)

    # Background — mean Z-score
    bg = Z.mean(axis=1).reshape(H, W)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8), facecolor="#0f1117")
    fig.suptitle(
        f"Total Deforestation Map\n{dates[0]} → {dates[-1]}",
        color="white", fontsize=15, fontweight="bold"
    )

    # Left: binary map
    ax = axes[0]
    ax.set_facecolor("#0f1117")
    # Green background for forest
    bg_norm = np.clip((bg + 3) / 6, 0, 1)
    ax.imshow(plt.cm.Greens(bg_norm)[:, :, :3], interpolation="bilinear")
    # Red anomaly pixels
    red_overlay = np.zeros((H, W, 4))
    red_overlay[total_anomaly] = [1.0, 0.15, 0.15, 0.9]
    ax.imshow(red_overlay, interpolation="nearest")
    ax.set_title("Where changes occurred", color="white", fontsize=12, pad=8)
    ax.axis("off")

    legend_els = [
        Patch(facecolor="#22c55e", label="Unchanged forest"),
        Patch(facecolor="#dc2626", label=f"Detected changes ({total_anomaly.sum():,} px)"),
    ]
    ax.legend(handles=legend_els, loc="lower right",
              facecolor="#1e2130", edgecolor="#444", labelcolor="white", fontsize=10)

    # Right: intensity (how many frames)
    ax2 = axes[1]
    ax2.set_facecolor("#0f1117")
    im2 = ax2.imshow(intensity, cmap="inferno", vmin=0,
                     vmax=max(intensity.max(), 1), interpolation="bilinear")
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.035, pad=0.02)
    cbar2.set_label("Number of frames with anomaly", color="white", fontsize=10)
    cbar2.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar2.ax.yaxis.get_ticklabels(), color="white")
    ax2.set_title("Change intensity (frame count)", color="white", fontsize=12, pad=8)
    ax2.axis("off")

    plt.tight_layout()
    out = VIDEO_DIR / "total_deforestation_map.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    print(f"[video] Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
def make_ndvi_change_map(Z, H, W, dates):
    """
    Z-score change map between the first and last frame.
    Shows where NDVI dropped the most.
    """
    delta = Z[:, -1] - Z[:, 0]
    delta_2d = delta.reshape(H, W)

    fig, ax = plt.subplots(figsize=(10, 8), facecolor="#0f1117")
    ax.set_facecolor("#0f1117")

    vmax = max(abs(delta_2d).max(), 0.1)
    im = ax.imshow(delta_2d, cmap="RdYlGn_r",
                   vmin=-vmax, vmax=vmax, interpolation="bilinear")

    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("ΔZ-score (negative = NDVI drop)", color="white", fontsize=10)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    ax.set_title(
        f"Z-score change: {dates[0]} → {dates[-1]}",
        color="white", fontsize=13, fontweight="bold", pad=12
    )
    ax.axis("off")

    plt.tight_layout()
    out = VIDEO_DIR / "ndvi_change_map.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    print(f"[video] Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
def run():
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    print("[video] Loading data...")
    Z, L, anomaly_mask, H, W, dates = load_data()

    # 1. Z-score GIF
    print("[video] Generating Z-score animation...")
    frames_z = make_zscore_frames(Z, H, W, dates)
    gif_path = VIDEO_DIR / "deforestation.gif"
    imageio.mimsave(str(gif_path), frames_z, fps=1, loop=0)
    print(f"[video] Saved: {gif_path}")

    # 2. Anomaly overlay GIF
    print("[video] Generating overlay animation...")
    frames_ov = make_overlay_frames(Z, anomaly_mask, H, W, dates)
    gif_ov_path = VIDEO_DIR / "deforestation_overlay.gif"
    imageio.mimsave(str(gif_ov_path), frames_ov, fps=1, loop=0)
    print(f"[video] Saved: {gif_ov_path}")

    # 3. Cumulative map
    print("[video] Generating cumulative map...")
    make_total_map(anomaly_mask, Z, H, W, dates)

    # 4. NDVI change map
    print("[video] Generating NDVI change map...")
    make_ndvi_change_map(Z, H, W, dates)

    print()
    print("[video] ✓ All files saved to output/video/")
    print(f"[video]   deforestation.gif")
    print(f"[video]   deforestation_overlay.gif")
    print(f"[video]   total_deforestation_map.png")
    print(f"[video]   ndvi_change_map.png")


if __name__ == "__main__":
    run()
