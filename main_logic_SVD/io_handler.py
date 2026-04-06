"""
IO handler for loading data from Max and saving results for Julia.

Input files (data/):
  X.npy      — matrix (pixels × frames)
  meta.npy   — dict: dates, height, width, bbox

Output files (output/):
  L.npy            — low-rank background
  S.npy            — anomalies (raw residuals, masked non-forest)
  Z.npy            — Z-score matrix
  anomaly_mask.npy — binary deforestation/fire mask
  output_meta.npy  — metadata for video (dates, sizes, thresholds)
"""

import numpy as np
from pathlib import Path


def load_input(data_dir: Path) -> tuple[np.ndarray, int, int, list]:
    """
    Loads X and metadata from Max.

    Returns
    X     : (pixels, frames)
    H, W  : dimensions of one image
    dates : list of strings 'YYYY-MM-DD'
    """
    X_path    = data_dir / "X.npy"
    meta_path = data_dir / "meta.npy"

    if not X_path.exists():
        raise FileNotFoundError(f"Not found {X_path} — ensure Max put files in data/")
    if not meta_path.exists():
        raise FileNotFoundError(f"Not found {meta_path}")

    X    = np.load(X_path)
    meta = np.load(meta_path, allow_pickle=True).item()

    H     = meta["height"]
    W     = meta["width"]
    dates = meta["dates"]

    print(f"[io]   X loaded: {X.shape}  ({H}×{W} px, {len(dates)} frames)")
    print(f"[io]   Period: {dates[0]}  →  {dates[-1]}")

    return X, H, W, dates


def save_outputs(
    output_dir: Path,
    L: np.ndarray,
    S: np.ndarray,
    Z: np.ndarray,
    anomaly_mask: np.ndarray,
    dates: list,
    H: int,
    W: int,
    z_threshold: float,
    ndvi_forest_threshold: float,
) -> None:
    """
    Saves all matrices for Julia.

    Output structure:
      L.npy            - stable background (for overlay in video)
      S.npy            - raw residuals
      Z.npy            - normalized anomaly intensity
      anomaly_mask.npy - True/False deforestation mask
      output_meta.npy  - all metadata in one dict
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "L.npy",            L)
    np.save(output_dir / "S.npy",            S)
    np.save(output_dir / "Z.npy",            Z)
    np.save(output_dir / "anomaly_mask.npy", anomaly_mask)
    np.save(output_dir / "output_meta.npy",  {
        "dates":                 dates,
        "height":                H,
        "width":                 W,
        "z_threshold":           z_threshold,
        "ndvi_forest_threshold": ndvi_forest_threshold,
    })

    print(f"[io]   Saved to {output_dir}/")
    print(f"[io]     L.npy            {L.shape}")
    print(f"[io]     S.npy            {S.shape}")
    print(f"[io]     Z.npy            {Z.shape}")
    print(f"[io]     anomaly_mask.npy {anomaly_mask.shape}")
    print(f"[io]     output_meta.npy")
    