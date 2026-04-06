"""
spatial_filter.py
─────────────────
Просторова фільтрація аномалій — відкидаємо точковий шум,
залишаємо тільки суцільні зв'язні плями (реальна вирубка).

Ідея:
  anomaly_mask містить True там де Z-score < threshold.
  Але хмари, тіні, шум дають ізольовані пікселі.
  Реальна вирубка — це СУЦІЛЬНА пляма з десятків/сотень пікселів.

  Алгоритм:
    1. Для кожного кадру розгортаємо anomaly_mask у 2D
    2. scipy.label() знаходить всі зв'язні компоненти (групи пікселів)
    3. Відкидаємо компоненти менші за MIN_CLUSTER_SIZE
    4. Результат — тільки великі суцільні плями
"""

import numpy as np
from scipy.ndimage import label, binary_dilation


# ── CONFIG ────────────────────────────────────────────────────────────────────
MIN_CLUSTER_SIZE = 10    # мінімум пікселів у зв'язній групі
DILATION_ITERS   = 1     # розширення маски перед кластеризацією
                         # (з'єднує майже-суцільні плями)


# ─────────────────────────────────────────────────────────────────────────────
def filter_spatial_noise(
    anomaly_mask: np.ndarray,
    H: int,
    W: int,
    min_cluster_size: int = MIN_CLUSTER_SIZE,
    dilation_iters:   int = DILATION_ITERS,
) -> tuple[np.ndarray, dict]:
    """
    Прибирає ізольовані пікселі з anomaly_mask.

    Parameters
    ----------
    anomaly_mask     : (pixels, frames) bool
    H, W             : розміри одного знімку
    min_cluster_size : мінімум пікселів у групі (менші — шум)
    dilation_iters   : скільки разів розширити маску перед кластеризацією

    Returns
    -------
    filtered_mask : (pixels, frames) bool — тільки великі плями
    stats         : dict з інформацією по кадрах
    """
    n_frames      = anomaly_mask.shape[1]
    filtered_mask = np.zeros_like(anomaly_mask)
    stats         = {"clusters_per_frame": [], "removed_noise_px": []}

    # структурний елемент — 8-зв'язність (діагоналі теж рахуємо)
    struct = np.ones((3, 3), dtype=bool)

    for t in range(n_frames):
        frame_mask = anomaly_mask[:, t].reshape(H, W)

        # Розширення — з'єднує пікселі що майже стикаються
        if dilation_iters > 0:
            dilated = binary_dilation(frame_mask, iterations=dilation_iters)
        else:
            dilated = frame_mask.copy()

        # Кластеризація зв'язних компонент
        labeled, n_components = label(dilated, structure=struct)

        # Залишаємо тільки великі кластери
        clean_frame   = np.zeros((H, W), dtype=bool)
        valid_clusters = 0

        for cluster_id in range(1, n_components + 1):
            cluster_pixels = (labeled == cluster_id)
            # Перетинаємо з оригінальною маскою (не розширеною)
            original_pixels = cluster_pixels & frame_mask
            if original_pixels.sum() >= min_cluster_size:
                clean_frame   |= original_pixels
                valid_clusters += 1

        noise_removed = frame_mask.sum() - clean_frame.sum()
        filtered_mask[:, t] = clean_frame.flatten()

        stats["clusters_per_frame"].append(valid_clusters)
        stats["removed_noise_px"].append(int(noise_removed))

    total_removed = sum(stats["removed_noise_px"])
    total_kept    = filtered_mask.sum()
    print(f"[spatial] MIN_CLUSTER_SIZE = {min_cluster_size} px")
    print(f"[spatial] Видалено шумових пікс : {total_removed}")
    print(f"[spatial] Залишено аномальних   : {total_kept}")
    print(f"[spatial] Кластерів по кадрах   : {stats['clusters_per_frame']}")

    return filtered_mask, stats
