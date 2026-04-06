"""
main_logic_SVD/pipeline.py
──────────────────────────
Пайплайн виявлення вирубки з єдиним методом:
SVD на baseline + регресійна детекція плавних змін.
"""

from pathlib import Path
import numpy as np

from main_logic_SVD.io_handler import load_input, save_outputs
from main_logic_SVD.svd_decomposition import compute_svd_background
from main_logic_SVD.forest_masks import build_forest_masks, NDVI_FOREST_THRESHOLD
from main_logic_SVD.change_detection import (
    compute_baseline,
    compute_regression_changes,
    BASELINE_WINDOW,
    VARIANCE_THRESHOLD,
    Z_THRESHOLD,
)
from main_logic_SVD.spatial_filter import filter_spatial_noise
from main_logic_SVD.diagnostics import plot_diagnostics


def run(data_dir: Path, output_dir: Path) -> None:
    """
    Єдиний пайплайн виявлення вирубки.
    """
    print("=" * 55)
    print("  Deforestation Pipeline [SVD + Regression]")
    print("=" * 55)

    X, H, W, dates = load_input(data_dir)
    forest_mask, nonforest_mask = build_forest_masks(X, H, W)

    baseline_window = BASELINE_WINDOW
    L_for_plot, _, sigma, k = compute_baseline(
        X,
        forest_mask,
        nonforest_mask,
        window=baseline_window,
        variance_threshold=VARIANCE_THRESHOLD,
    )

    anomaly_mask, Z, change_time = compute_regression_changes(
        X,
        forest_mask,
        nonforest_mask,
        H,
        W,
        baseline_window=baseline_window,
        verbose=True,
    )

    S_masked = X - L_for_plot
    anomaly_clean, _ = filter_spatial_noise(anomaly_mask, H, W)

    plot_diagnostics(
        sigma,
        X,
        L_for_plot,
        Z,
        anomaly_clean,
        H,
        W,
        dates,
        k,
        output_dir,
        title_suffix="SVD + Regression",
    )

    save_outputs(
        output_dir=output_dir,
        L=L_for_plot,
        S=S_masked,
        Z=Z,
        anomaly_mask=anomaly_clean,
        dates=dates,
        H=H,
        W=W,
        z_threshold=Z_THRESHOLD,
        ndvi_forest_threshold=NDVI_FOREST_THRESHOLD,
    )

    print("=" * 55)
    print(f"  Готово. Результати → {output_dir}/")
    print("=" * 55)
