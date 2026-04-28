# convert_dataset_to_images.py
#
# Перетворює X.npy у набір PNG кадрів.
#
# Приклад:
# python3 convert_dataset_to_images.py \
#   --input tests/test_images \
#   --output exported_frames
#
# Результат:
# exported_frames/
#   gradual_middle/
#       frame_000.png
#       frame_001.png
#   sudden_big/
#       frame_000.png
#       ...

from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt


def export_one_scenario(x_path: Path, output_root: Path):
    scenario_name = x_path.parent.name
    scenario_out = output_root / scenario_name
    scenario_out.mkdir(parents=True, exist_ok=True)

    X = np.load(x_path)

    print(f"\n=== {scenario_name} ===")
    print("X shape:", X.shape)

    # ----------------------------------
    # format (T,H,W)
    # ----------------------------------
    if X.ndim == 3:
        T, H, W = X.shape

        for t in range(T):
            frame = X[t]

            plt.imsave(
                scenario_out / f"frame_{t:03d}.png",
                frame,
                cmap="RdYlGn",
                vmin=0.0,
                vmax=1.0
            )

        print(f"Saved {T} frames -> {scenario_out}")

    # ----------------------------------
    # format (N,T)
    # ----------------------------------
    elif X.ndim == 2:
        N, T = X.shape

        side = int(np.sqrt(N))
        if side * side != N:
            raise ValueError(f"{scenario_name}: N={N} not square")

        H = W = side

        for t in range(T):
            frame = X[:, t].reshape(H, W)

            plt.imsave(
                scenario_out / f"frame_{t:03d}.png",
                frame,
                cmap="RdYlGn",
                vmin=0.0,
                vmax=1.0
            )

        print(f"Saved {T} frames ({H}x{W}) -> {scenario_out}")

    else:
        print("Unsupported shape:", X.shape)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="tests/test_images",
        help="де шукати сценарії з X.npy"
    )

    parser.add_argument(
        "--output",
        default="exported_frames",
        help="куди зберігати PNG кадри"
    )

    parser.add_argument(
        "--scenario",
        default=None,
        help="назва одного сценарію (наприклад gradual_middle)"
    )

    args = parser.parse_args()

    input_root = Path(args.input)
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    all_x = list(input_root.rglob("X.npy"))

    if args.scenario:
        all_x = [p for p in all_x if p.parent.name == args.scenario]

    if not all_x:
        print("No X.npy found")
        return

    for x_path in all_x:
        export_one_scenario(x_path, output_root)


if __name__ == "__main__":
    main()