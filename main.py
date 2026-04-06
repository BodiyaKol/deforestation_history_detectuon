"""
main.py
───────
Кореневий файл проєкту. Запускай саме його.

  python main.py

Потік:
  data_input_to_matrix/input_logic.py  →  data/X.npy + data/meta.npy
  main_logic_SVD/pipeline.py           →  output/L, S, Z, anomaly_mask
  convert_to_video/                    ←  (Юлія) читає output/ та рендерить відео
"""

from pathlib import Path
import sys

# ── Шляхи ────────────────────────────────────────────────────────────────────
DATA_DIR   = Path("data")
OUTPUT_DIR = Path("output")


# ── Крок 1: підготовка даних (Максим) ────────────────────────────────────────
def step_prepare_data() -> None:
    """
    Запускає data_input_to_matrix/input_logic.py.
    Результат: data/X.npy та data/meta.npy
    """
    print("\n[STEP 1]  Підготовка даних — input_logic")
    print("-" * 45)

    # Перевіряємо чи вже є готові файли
    if (DATA_DIR / "X.npy").exists() and (DATA_DIR / "meta.npy").exists():
        print("[skip]  data/X.npy вже існує — пропускаємо завантаження")
        print("[tip]   Щоб перезавантажити — видали data/X.npy та запусти знову")
        return

    DATA_DIR.mkdir(exist_ok=True)

    # Запускаємо логіку Максима
    import importlib.util, os
    spec = importlib.util.spec_from_file_location(
        "input_logic",
        Path("data_input_to_matrix") / "input_logic.py"
    )
    module = importlib.util.module_from_spec(spec)

    # input_logic.py зберігає файли відносно поточної директорії —
    # тимчасово переходимо в корінь проєкту (вже там)
    spec.loader.exec_module(module)
    print("[ok]    data/X.npy та data/meta.npy збережено")


# ── Крок 2: SVD-пайплайн (Богдан) ────────────────────────────────────────────
def step_svd_pipeline() -> None:
    """
    Запускає main_logic_SVD/pipeline.py.
    Результат: output/L, S, Z, anomaly_mask, output_meta
    """
    print("\n[STEP 2]  SVD-пайплайн — main_logic_SVD")
    print("-" * 45)

    from main_logic_SVD import run
    run(data_dir=DATA_DIR, output_dir=OUTPUT_DIR)


# ── Крок 3: Відео (Юлія) — placeholder ───────────────────────────────────────
def step_render_video() -> None:
    """
    Запускає convert_to_video/ (Юлія).
    Читає output/ та рендерить фінальне відео.
    """
    print("\n[STEP 3]  Рендер відео — convert_to_video")
    print("-" * 45)

    video_script = Path("convert_to_video") / "render.py"
    if not video_script.exists():
        print("[skip]  convert_to_video/render.py не знайдено — Юлія ще не готова")
        return

    import importlib.util
    spec   = importlib.util.spec_from_file_location("render", video_script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        step_prepare_data()
        step_svd_pipeline()
        step_render_video()
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)