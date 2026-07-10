# regenerate_processed_data.py
# ============================================================================
# WHY THIS EXISTS:
# Reruns the exact same preprocessing logic as preprocess.py (MinMax scaling,
# 60-step lookback windowing, same train/val/test split ratios) but points
# it at the new 18,000-candle extended dataset instead of the original
# 1,000-candle one. This keeps everything downstream (environments, training
# scripts) unchanged — only the DATA gets bigger, nothing else changes, so
# any improvement in results can be attributed cleanly to having more data.
#
# The original small X.npy/y.npy are backed up first, not overwritten, in
# case you want to reference or compare against them later.
# ============================================================================

import pandas as pd
import numpy as np
import os
import shutil

from preprocess import preprocess_data  # reuse your exact original logic, unchanged

EXTENDED_CSV = "data/raw/binance_btcusdt_extended.csv"
BACKUP_DIR = "data/processed/original_small_backup"


def backup_original_arrays():
    os.makedirs(BACKUP_DIR, exist_ok=True)
    files_to_backup = [
        "X.npy", "y.npy", "X_train.npy", "y_train.npy",
        "X_val.npy", "y_val.npy", "X_test.npy", "y_test.npy",
    ]
    backed_up = []
    for fname in files_to_backup:
        src = f"data/processed/{fname}"
        if os.path.exists(src):
            shutil.copy(src, f"{BACKUP_DIR}/{fname}")
            backed_up.append(fname)
    if backed_up:
        print(f"Backed up original arrays to {BACKUP_DIR}/: {', '.join(backed_up)}")
    else:
        print("No existing arrays found to back up (fresh run).")


def main():
    print("=" * 80)
    print("REGENERATING PROCESSED DATA FROM EXTENDED (2-YEAR) DATASET")
    print("=" * 80)

    backup_original_arrays()

    print(f"\nLoading extended dataset: {EXTENDED_CSV}")
    df = pd.read_csv(EXTENDED_CSV)
    print(f"Raw extended data: {df.shape}")

    # Keep only OHLCV, same as original preprocess.py — timestamp dropped
    # inside preprocess_data() itself, so no change needed here.
    print("\nRunning preprocessing (60-step lookback, MinMax scaling, 70/15/15 split)...")
    preprocess_data(df, lookback=60, forecast_horizon=1)

    # Verify the results
    X = np.load("data/processed/X.npy")
    y = np.load("data/processed/y.npy")
    X_train = np.load("data/processed/X_train.npy")
    X_test = np.load("data/processed/X_test.npy")

    print("\n" + "=" * 80)
    print("VERIFICATION")
    print("=" * 80)
    print(f"Full dataset:  X {X.shape}, y {y.shape}")
    print(f"Train split:   X_train {X_train.shape}")
    print(f"Test split:    X_test {X_test.shape}")
    print(f"\nCompare to original: X was (939, 60, 5) -> now ({X.shape[0]}, {X.shape[1]}, {X.shape[2]})")
    print(f"That's a {X.shape[0] / 939:.1f}x increase in sample count.")


if __name__ == "__main__":
    main()
