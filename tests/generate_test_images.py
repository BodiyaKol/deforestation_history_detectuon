"""
tests/generate_test_images.py
──────────────────────────────
Генерує синтетичні супутникові знімки для тестування пайплайну.

Використання:
  python tests/generate_test_images.py --resolution low
  python tests/generate_test_images.py --resolution middle
  python tests/generate_test_images.py --resolution big
  python tests/generate_test_images.py --resolution low --frames 8
  python tests/generate_test_images.py --resolution low --scenario gradual
  python tests/generate_test_images.py --all   ← генерує всі сценарії

Сценарії (--scenario):
  sudden    — раптова вирубка (1–2 кадри, різке падіння NDVI)
  gradual   — поступова деградація (пожежа, повільне всихання)
  scattered — розсіяні маленькі вирубки (тест на шум vs сигнал)
  clean     — без змін (для перевірки false positives)
  mixed     — комбінація sudden + scattered (реалістично)

Роздільна здатність (--resolution):
  low    →  30×30 px,  8 кадрів
  middle →  60×60 px,  12 кадрів
  big    → 100×100 px, 20 кадрів

Структура виводу:
  tests/test_images/<scenario>_<resolution>/
    X.npy
    meta.npy
    ground_truth.npy    ← (H, W, T) bool — де реальна вирубка
    scenario_info.txt   ← опис сценарію
"""

import numpy as np
import argparse
from pathlib import Path


# ── Конфіг роздільних здатностей ─────────────────────────────────────────────
RESOLUTIONS = {
    "low":    {"H": 30,  "W": 30,  "T_default": 8},
    "middle": {"H": 60,  "W": 60,  "T_default": 12},
    "big":    {"H": 100, "W": 100, "T_default": 20},
}

# ── Базові параметри сцени ─────────────────────────────────────────────────────
FOREST_NDVI    = 0.65
NONFOREST_NDVI = 0.12
NOISE_STD      = 0.010
SEASONAL_AMP   = 0.04


# ─────────────────────────────────────────────────────────────────────────────
#  Базова сцена
# ─────────────────────────────────────────────────────────────────────────────
def _make_base(H: int, W: int, T: int, seed: int = 0) -> tuple:
    """
    Базова сцена: ліс всередині, поле по краях.
    Повертає X (pixels×T) та base_2d (H×W).
    """
    rng = np.random.default_rng(seed)
    base = np.full((H, W), NONFOREST_NDVI)
    margin = max(2, H // 6)
    base[margin:H-margin, margin:W-margin] = FOREST_NDVI

    seasonal = SEASONAL_AMP * np.sin(2 * np.pi * np.arange(T) / T)
    frames   = [base + seasonal[t] + rng.normal(0, NOISE_STD, (H, W))
                for t in range(T)]
    X = np.stack(frames).reshape(T, H * W).T
    return np.clip(X, -1.0, 1.0), base


def _dates(T: int) -> list:
    """Генерує список дат (по ~2 тижні)."""
    from datetime import date, timedelta
    start = date(2024, 1, 1)
    return [(start + timedelta(days=14*t)).strftime("%Y-%m-%d") for t in range(T)]


def _save(out_dir: Path, X, H, W, T, gt_3d, info_lines):
    """Зберігає X.npy, meta.npy, ground_truth.npy, scenario_info.txt."""
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "X.npy", X)
    np.save(out_dir / "meta.npy", {
        "dates": _dates(T), "height": H, "width": W,
        "bbox": [23.55, 48.50, 23.60, 48.55],
    })
    np.save(out_dir / "ground_truth.npy", gt_3d)   # (H, W, T) bool
    (out_dir / "scenario_info.txt").write_text("\n".join(info_lines))
    print(f"  → {out_dir}  [{H}×{W}, {T} кадрів]")


# ─────────────────────────────────────────────────────────────────────────────
#  Сценарії
# ─────────────────────────────────────────────────────────────────────────────
def scenario_sudden(H, W, T, seed=1) -> tuple:
    """
    Раптова вирубка: прямокутна ділянка, NDVI різко падає до 0.03
    протягом 1 кадру і залишається низьким.
    Ground truth: True у зоні вирубки для кадрів після події.
    """
    X, base = _make_base(H, W, T, seed)
    rng     = np.random.default_rng(seed + 10)
    margin  = max(2, H // 6)

    # Зона вирубки: 20% площі лісу
    size_r  = max(3, (H - 2*margin) // 3)
    size_c  = max(3, (W - 2*margin) // 3)
    r0      = margin + 2
    c0      = margin + 2
    event_t = T // 2   # кадр події

    gt_2d = np.zeros((H, W), dtype=bool)
    gt_2d[r0:r0+size_r, c0:c0+size_c] = True
    flat  = gt_2d.flatten()

    for t in range(event_t, T):
        X[flat, t] = 0.03 + rng.normal(0, 0.005, flat.sum())

    gt_3d = np.zeros((H, W, T), dtype=bool)
    for t in range(event_t, T):
        gt_3d[:, :, t] = gt_2d

    info = [
        "SCENARIO: sudden deforestation",
        f"  Size: {H}x{W}, T={T}",
        f"  Deforestation zone: rows {r0}:{r0+size_r}, cols {c0}:{c0+size_c}",
        f"  Event frame: {event_t}",
        f"  Target NDVI: 0.03",
    ]
    return X, gt_3d, info


def scenario_seasonal(H, W, T, seed=5) -> tuple:
    """
    Сезонність: сильні природні коливання NDVI без аномалій.
    Демонструє, як SVD моделює сезонні тренди у baseline періоді.
    NDVI коливається від 0.55 до 0.75 з амплітудою 0.1.
    """
    rng = np.random.default_rng(seed)
    base = np.full((H, W), NONFOREST_NDVI)
    margin = max(2, H // 6)
    base[margin:H-margin, margin:W-margin] = FOREST_NDVI

    # Сильна сезонність: амплітуда 0.1 замість 0.04
    seasonal = 0.1 * np.sin(2 * np.pi * np.arange(T) / T)
    frames   = [base + seasonal[t] + rng.normal(0, NOISE_STD, (H, W))
                for t in range(T)]
    X = np.stack(frames).reshape(T, H * W).T
    X = np.clip(X, -1.0, 1.0)

    # Ніяких аномалій — чиста сезонність
    gt_3d = np.zeros((H, W, T), dtype=bool)

    info = [
        "SCENARIO: seasonal variation",
        f"  Size: {H}x{W}, T={T}",
        f"  Seasonal amplitude: 0.1 (strong)",
        f"  No anomalies — demonstrates SVD baseline modeling",
        f"  Forest NDVI range: ~{FOREST_NDVI-0.1:.2f} to {FOREST_NDVI+0.1:.2f}",
    ]
    return X, gt_3d, info


def scenario_gradual(H, W, T, seed=2) -> tuple:
    """
    Поступова деградація (пожежа, посуха): NDVI знижується лінійно
    від нормального до 0.1 протягом останніх T//2 кадрів.
    """
    X, base = _make_base(H, W, T, seed)
    rng     = np.random.default_rng(seed + 20)
    margin  = max(2, H // 6)

    size_r  = max(3, (H - 2*margin) // 3)
    size_c  = max(3, (W - 2*margin) // 3)
    r0, c0  = margin + 2, margin + (W - 2*margin) // 2
    start_t = T // 2

    gt_2d   = np.zeros((H, W), dtype=bool)
    gt_2d[r0:r0+size_r, c0:c0+size_c] = True
    flat    = gt_2d.flatten()
    n_px    = flat.sum()

    for t in range(start_t, T):
        progress    = (t - start_t) / max(1, T - start_t)
        target_ndvi = FOREST_NDVI - progress * (FOREST_NDVI - 0.10)
        X[flat, t]  = target_ndvi + rng.normal(0, 0.008, n_px)

    gt_3d = np.zeros((H, W, T), dtype=bool)
    for t in range(start_t + (T - start_t) // 2, T):   # позначаємо тільки суттєві кадри
        gt_3d[:, :, t] = gt_2d

    info = [
        "SCENARIO: gradual degradation (fire/drought)",
        f"  Size: {H}x{W}, T={T}",
        f"  Degradation zone: rows {r0}:{r0+size_r}, cols {c0}:{c0+size_c}",
        f"  Starts at frame {start_t}, NDVI: {FOREST_NDVI:.2f} → 0.10",
    ]
    return X, gt_3d, info


def scenario_scattered(H, W, T, seed=3) -> tuple:
    """
    Розсіяні маленькі вирубки: кілька маленьких плям (3×3 px).
    Тест чи алгоритм відрізняє реальні кластери від шуму.
    """
    X, base = _make_base(H, W, T, seed)
    rng     = np.random.default_rng(seed + 30)
    margin  = max(2, H // 6)
    event_t = T // 2

    gt_2d   = np.zeros((H, W), dtype=bool)
    n_spots = 4
    spot_sz = max(2, H // 15)

    spots_placed = []
    attempts     = 0
    while len(spots_placed) < n_spots and attempts < 200:
        r = rng.integers(margin + 1, H - margin - spot_sz - 1)
        c = rng.integers(margin + 1, W - margin - spot_sz - 1)
        # не перекриватись
        ok = all(abs(r - pr) > spot_sz + 2 and abs(c - pc) > spot_sz + 2
                 for pr, pc in spots_placed)
        if ok:
            gt_2d[r:r+spot_sz, c:c+spot_sz] = True
            spots_placed.append((r, c))
        attempts += 1

    flat = gt_2d.flatten()
    for t in range(event_t, T):
        X[flat, t] = 0.03 + rng.normal(0, 0.005, flat.sum())

    gt_3d = np.zeros((H, W, T), dtype=bool)
    for t in range(event_t, T):
        gt_3d[:, :, t] = gt_2d

    info = [
        "SCENARIO: scattered small clearcuts",
        f"  Size: {H}x{W}, T={T}",
        f"  Spots placed: {len(spots_placed)}, size: {spot_sz}x{spot_sz}",
        f"  Event frame: {event_t}",
        f"  Spot locations: {spots_placed}",
    ]
    return X, gt_3d, info


def scenario_clean(H, W, T, seed=4) -> tuple:
    """
    Без змін — тест на false positives.
    Ground truth = всі нулі.
    """
    X, base = _make_base(H, W, T, seed)
    gt_3d   = np.zeros((H, W, T), dtype=bool)

    info = [
        "SCENARIO: clean (no deforestation)",
        f"  Size: {H}x{W}, T={T}",
        "  Expected: zero anomalies (false positive test)",
    ]
    return X, gt_3d, info


def scenario_mixed(H, W, T, seed=5) -> tuple:
    """
    Комбінація: одна велика раптова вирубка + кілька маленьких.
    Реалістичний сценарій.
    """
    X1, gt1, _ = scenario_sudden(H, W, T, seed=seed)
    X2, gt2, _ = scenario_scattered(H, W, T, seed=seed + 100)

    # Накладаємо маленькі вирубки поверх великої
    mask_diff = gt2 & ~gt1   # тільки нові пікселі
    flat = mask_diff.any(axis=2).flatten()
    rng  = np.random.default_rng(seed + 50)
    event_t = T // 2
    for t in range(event_t, T):
        X1[flat, t] = 0.03 + rng.normal(0, 0.005, flat.sum())

    gt_3d = gt1 | gt2

    info = [
        "SCENARIO: mixed (large + scattered clearcuts)",
        f"  Size: {H}x{W}, T={T}",
        "  Combined: sudden large + scattered small deforestation",
    ]
    return X1, gt_3d, info


# ─────────────────────────────────────────────────────────────────────────────
SCENARIOS = {
    "sudden":    scenario_sudden,
    "gradual":   scenario_gradual,
    "scattered": scenario_scattered,
    "clean":     scenario_clean,
    "mixed":     scenario_mixed,
    "seasonal":  scenario_seasonal,
}

# ─────────────────────────────────────────────────────────────────────────────
def generate(scenario: str, resolution: str, frames: int | None, out_base: Path):
    cfg  = RESOLUTIONS[resolution]
    H, W = cfg["H"], cfg["W"]
    T    = frames if frames else cfg["T_default"]

    fn      = SCENARIOS[scenario]
    X, gt3d, info = fn(H, W, T)

    out_dir = out_base / f"{scenario}_{resolution}"
    _save(out_dir, X, H, W, T, gt3d, info)


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Generate synthetic test images")
    parser.add_argument("--resolution", choices=["low","middle","big"], default="low")
    parser.add_argument("--scenario",   choices=list(SCENARIOS.keys()), default="sudden")
    parser.add_argument("--frames",     type=int, default=None,
                        help="Override number of frames")
    parser.add_argument("--all",        action="store_true",
                        help="Generate all scenarios × all resolutions")
    parser.add_argument("--out",        default="tests/test_images",
                        help="Output base directory")
    args = parser.parse_args()

    out_base = Path(args.out)
    print(f"[gen]  Output → {out_base}/")

    if args.all:
        for sc in SCENARIOS:
            for res in RESOLUTIONS:
                generate(sc, res, args.frames, out_base)
    else:
        generate(args.scenario, args.resolution, args.frames, out_base)

    print("[gen]  Готово.")


if __name__ == "__main__":
    main()
    