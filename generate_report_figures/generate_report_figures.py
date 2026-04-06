import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    # Налаштування шляхів
    data_dir = Path("data")
    output_dir = Path("output")
    fig_dir = Path("figures")
    fig_dir.mkdir(exist_ok=True) # Створюємо папку figures/, якщо її немає

    print("Завантаження даних...")
    try:
        X = np.load(data_dir / "X.npy")
        L = np.load(output_dir / "L.npy")
        mask = np.load(output_dir / "anomaly_mask.npy")
        meta = np.load(output_dir / "output_meta.npy", allow_pickle=True).item()
    except FileNotFoundError as e:
        print(f"❌ Помилка: не знайдено файл {e.filename}. Запусти спочатку main.py!")
        return

    H, W = meta["height"], meta["width"]
    dates = meta["dates"]
    N = H * W
    T = len(dates)

    # 1. Дані для тексту (статистика)
    sorted_dates = sorted(dates)
    start_date = sorted_dates[0]
    end_date = sorted_dates[-1]

    # Шукаємо кадр з найбільшою кількістю аномалій (найбільша вирубка)
    anomalies_per_frame = mask.sum(axis=0)
    best_idx = np.argmax(anomalies_per_frame)
    peak_date = dates[best_idx]
    peak_anomalies = anomalies_per_frame[best_idx]
    peak_percent = (peak_anomalies / N) * 100

    # Сумарні аномалії
    cumulative_mask = mask.any(axis=1)
    cumulative_anomalies = cumulative_mask.sum()
    cumulative_percent = (cumulative_anomalies / N) * 100

    def reconstruct(arr_1d):
        """Відновлює розмір HxW (враховує, якщо видалялися якісь пікселі)"""
        if len(arr_1d) == N: return arr_1d.reshape(H, W)
        pad = np.zeros(N); pad[:len(arr_1d)] = arr_1d
        return pad.reshape(H, W)

    # ---------------------------------------------------------
    # FIGURE 1: Scree Plot (Швидко рахуємо SVD для X)
    print("Генерація Figure 1: scree_plot.png...")
    # Оскільки T дуже маленьке (наприклад, 6 дат), SVD порахується миттєво
    _, s, _ = np.linalg.svd(X, full_matrices=False)
    
    plt.figure(figsize=(7, 5))
    plt.plot(range(1, len(s) + 1), s, 'o-', color='#1f77b4', linewidth=2.5, markersize=8)
    plt.xlabel('Component index (k)', fontsize=12)
    plt.ylabel('Singular value (σ)', fontsize=12)
    plt.title('Scree Plot of Matrix X', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(fig_dir / "scree_plot.png", dpi=200, bbox_inches='tight')
    plt.close()

    # ---------------------------------------------------------
    # FIGURE 2: Background L
    print("Генерація Figure 2: background_L.png...")
    bg_frame = reconstruct(L[:, 0]) # Беремо першу колонку фону
    
    plt.figure(figsize=(9, 6))
    im = plt.imshow(bg_frame, cmap='RdYlGn', vmin=0, vmax=0.9)
    plt.title('Reconstructed Forest Background (Low-Rank L)', fontsize=14)
    plt.axis('off')
    plt.colorbar(im, fraction=0.03, pad=0.04, label="NDVI")
    plt.savefig(fig_dir / "background_L.png", dpi=200, bbox_inches='tight')
    plt.close()

    # ---------------------------------------------------------
    # FIGURE 3: Anomaly Mask для найцікавішої дати
    print(f"Генерація Figure 3: anomaly_mask.png (для дати {peak_date})...")
    bg_best = reconstruct(L[:, best_idx])
    mask_best = reconstruct(mask[:, best_idx])
    
    # Робимо фон чорно-білим для контрасту
    bg_norm = (bg_best - np.nanmin(bg_best)) / (np.nanmax(bg_best) - np.nanmin(bg_best) + 1e-8)
    
    plt.figure(figsize=(9, 6))
    plt.imshow(bg_norm, cmap='gray', alpha=0.7)
    
    overlay = np.zeros((H, W, 4))
    overlay[mask_best == True] = [1, 0, 0, 1] # Яскраво червоний
    plt.imshow(overlay)
    
    plt.title(f'Deforestation Anomaly Mask (Date: {peak_date})', fontsize=14)
    plt.axis('off')
    plt.savefig(fig_dir / "anomaly_mask.png", dpi=200, bbox_inches='tight')
    plt.close()

    # ---------------------------------------------------------
    # FIGURE 4: Cumulative Map
    print("Генерація Figure 4: cumulative_map.png...")
    mask_cum = reconstruct(cumulative_mask)
    
    plt.figure(figsize=(9, 6))
    plt.imshow(bg_norm, cmap='gray', alpha=0.5)
    
    overlay_cum = np.zeros((H, W, 4))
    overlay_cum[mask_cum == True] = [1, 0, 0, 1]
    plt.imshow(overlay_cum)
    
    plt.title('Cumulative Deforestation Map (All Dates)', fontsize=14)
    plt.axis('off')
    plt.savefig(fig_dir / "cumulative_map.png", dpi=200, bbox_inches='tight')
    plt.close()

    # ---------------------------------------------------------
    # ВИВІД СТАТИСТИКИ ДЛЯ ЗВІТУ
    print("\n" + "="*60)
    print("✅ ГОТОВО! Всі картинки лежать у папці 'figures/'.")
    print("Завантаж цю папку в Overleaf.")
    print("="*60)
    print("📊 ДАНІ ДЛЯ ВСТАВКИ В LATEX (СЕКЦІЯ 6):\n")
    print(f"[N] acquisitions         -> {T}")
    print(f"[start date]             -> {start_date}")
    print(f"[end date]               -> {end_date}")
    print(f"[geographic area]        -> Drohobych region (або просто 'Lviv region')")
    print(f"Total pixels [N]         -> {N:,}")
    print(f"Peak anomaly pixels      -> {peak_anomalies:,} ({peak_percent:.2f}%)")
    print(f"Cumulative anomaly px    -> {cumulative_anomalies:,} ({cumulative_percent:.2f}%)")
    print("="*60)
    print(f"💡 ПІДКАЗКА: В описі до Figure 3 (де [date]) впиши дату: {peak_date}")

if __name__ == "__main__":
    main()