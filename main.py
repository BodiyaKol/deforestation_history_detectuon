"""
Main project file.
"""

from pathlib import Path
import sys
import argparse

DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")


def step_prepare_data() -> None:
    print("\n[STEP 1] Data preparation — input_logic")
    print("-" * 45)
    if (DATA_DIR / "X.npy").exists() and (DATA_DIR / "meta.npy").exists():
        print("[skip] data/X.npy exists — skipping download")
        print("[tip] To reload — delete data/X.npy and run again")
        return

    DATA_DIR.mkdir(exist_ok=True)
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "input_logic",
        Path("data_input_to_matrix") / "input_logic.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    print("[ok] data/X.npy and data/meta.npy saved")


def step_pipeline() -> None:
    print("\n[STEP 2] Deforestation detection pipeline")
    print("-" * 45)
    from main_logic_SVD import run
    run(data_dir=DATA_DIR, output_dir=OUTPUT_DIR)


def step_render_video() -> None:
    print("\n[STEP 3] Render video — convert_to_video")
    print("-" * 45)
    video_script = Path("convert_to_video") / "render.py"
    if not video_script.exists():
        print("[skip] convert_to_video/render.py not found — Julia not ready yet")
        return
    import importlib.util
    spec = importlib.util.spec_from_file_location("render", video_script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Forest deforestation detection")
    parser.add_argument("--skip-data", action="store_true", help="Skip step 1 (data preparation)")
    parser.add_argument("--skip-video", action="store_true", help="Skip step 3 (render video)")
    args = parser.parse_args()

    try:
        if not args.skip_data:
            step_prepare_data()
        step_pipeline()
        if not args.skip_video:
            step_render_video()
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
