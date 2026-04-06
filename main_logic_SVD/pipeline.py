"""
main_logic_SVD/pipeline.py
──────────────────────────
Точка входу для SVD-пайплайну.
Склеює всі модулі в єдиний потік: load → SVD → masks → anomalies → save.

Запускається з кореневого main.py — не запускай цей файл напряму.
"""

from pathlib import Path

from main_logic_SVD.io_handler        import load_input, save_outputs
from main_logic_SVD.svd_decomposition import compute_svd_background
from main_logic_SVD.forest_masks      import build_forest_masks, NDVI_FOREST_THRESHOLD
from main_logic_SVD.anomaly_detection import compute_anomalies, ANOMALY_Z_THRESHOLD
from main_logic_SVD.spatial_filter    import filter_spatial_noise
from main_logic_SVD.diagnostics       import plot_diagnostics


def run(data_dir: Path, output_dir: Path) -> None:
    """
    Повний SVD-пайплайн виявлення вирубки.

    Кроки:
      1. load       — завантаження X та метаданих
      2. SVD        — побудова фону L та залишків S
      3. masks      — маски лісу/не-лісу за першим знімком
      4. anomalies  — Z-score + порогування → anomaly_mask
      5. spatial    — прибираємо точковий шум, залишаємо суцільні плями
      6. diagnostics
      7. save
    """
    print("=" * 55)
    print("  SVD Deforestation Pipeline")
    print("=" * 55)

    X, H, W, dates              = load_input(data_dir)
    L, S, sigma, k              = compute_svd_background(X)
    forest_mask, nonforest_mask = build_forest_masks(X, H, W)
    S_masked, Z, anomaly_mask   = compute_anomalies(S, forest_mask, nonforest_mask, H, W)

    print()
    anomaly_clean, _ = filter_spatial_noise(anomaly_mask, H, W)

    plot_diagnostics(sigma, X, L, Z, anomaly_clean, H, W, dates, k, output_dir)

    save_outputs(
        output_dir            = output_dir,
        L                     = L,
        S                     = S_masked,
        Z                     = Z,
        anomaly_mask          = anomaly_clean,
        dates                 = dates,
        H                     = H,
        W                     = W,
        z_threshold           = ANOMALY_Z_THRESHOLD,
        ndvi_forest_threshold = NDVI_FOREST_THRESHOLD,
    )

    print("=" * 55)
    print(f"  Готово. Результати → {output_dir}/")
    print("=" * 55)
    