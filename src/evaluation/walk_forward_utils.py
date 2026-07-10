# walk_forward_utils.py
# ============================================================================
# WHY THIS EXISTS:
# The original evaluation used a single chronological train/test split
# (85/15), tested on ~140-150 timesteps. Any single-window backtest is
# highly sensitive to which specific period you happened to test on — a
# handful of lucky trades can dominate a small sample and produce an
# unrealistic Sharpe ratio (this is exactly what debug_environments.py
# flagged in the original project).
#
# This module implements ANCHORED (expanding-window) walk-forward
# validation: the training set grows over time, and each subsequent test
# window is a fresh, never-before-seen chronological slice. This gives
# multiple independent out-of-sample evaluations instead of one, so we can
# report mean +/- std of every metric across windows — a far more credible
# claim than a single point estimate.
#
#   Window 1: train [0        : T0]        test [T0       : T0+test_size]
#   Window 2: train [0        : T0+step]   test [T0+step  : T0+step+test_size]
#   Window 3: train [0        : T0+2*step] test [...]
#   ...
# ============================================================================

import numpy as np


def generate_walk_forward_windows(total_length, n_windows=5, initial_train_frac=0.5, test_frac=0.1):
    """
    Generate anchored (expanding-window) walk-forward train/test index pairs.

    Args:
        total_length: total number of samples in the full chronological dataset
        n_windows: number of walk-forward folds to generate
        initial_train_frac: fraction of data used for the FIRST training window
        test_frac: fraction of total data used as each test window's size

    Returns:
        List of dicts: [{'window': i, 'train_start':.., 'train_end':.., 'test_start':.., 'test_end':..}, ...]
    """
    test_size = int(total_length * test_frac)
    initial_train_end = int(total_length * initial_train_frac)

    # Space remaining after the initial training block, divided across windows
    remaining = total_length - initial_train_end
    step = max(test_size, remaining // n_windows)

    windows = []
    train_end = initial_train_end

    for i in range(n_windows):
        test_start = train_end
        test_end = min(test_start + test_size, total_length)

        if test_start >= total_length or test_end - test_start < 10:
            break  # not enough data left for a meaningful test window

        windows.append({
            "window": i + 1,
            "train_start": 0,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
        })

        train_end = min(train_end + step, total_length)

    return windows


def summarize_windows(windows, total_length):
    """Pretty-print window sizes for sanity-checking before a long training run."""
    print(f"Total dataset length: {total_length}")
    print(f"Generated {len(windows)} walk-forward windows:\n")
    for w in windows:
        train_size = w["train_end"] - w["train_start"]
        test_size = w["test_end"] - w["test_start"]
        print(f"  Window {w['window']}: train=[0:{w['train_end']}] ({train_size} samples) "
              f"-> test=[{w['test_start']}:{w['test_end']}] ({test_size} samples)")


if __name__ == "__main__":
    # Quick sanity check with a dummy length
    windows = generate_walk_forward_windows(total_length=1000, n_windows=5)
    summarize_windows(windows, total_length=1000)
