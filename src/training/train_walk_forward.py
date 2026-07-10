# train_walk_forward.py
# ============================================================================
# DAY 1-2 DELIVERABLE: Walk-forward validation + reward function ablation
#
# WHY THIS EXISTS:
# The original project trained once, on one train/test split, and reported
# a single Sharpe/return number. This script trains and evaluates across:
#
#   - Multiple walk-forward windows (independent out-of-sample periods)
#   - Multiple reward function designs (aggressive / raw_return / differential_sharpe)
#   - Multiple random seeds per (window, reward_mode) combination
#
# ...and saves every run's metrics to disk so evaluate_walk_forward.py can
# compute mean +/- std per configuration. This directly answers "are your
# results realistic?" with evidence instead of a single lucky number.
#
# USAGE (run this on Colab/Kaggle with GPU runtime):
#   python train_walk_forward.py
#
# Expects data/processed/X.npy and data/processed/y.npy to exist (the FULL
# chronological arrays produced by preprocess.py, BEFORE the train/test
# split) — walk-forward windows are carved out of this full timeline.
# ============================================================================

import os
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

from updated_rl_env_v2 import BitcoinTradingEnv
from walk_forward_utils import generate_walk_forward_windows, summarize_windows
from portfolio_manager_v2 import REWARD_FORMULA_VERSION


def format_duration(seconds):
    """Human-readable duration: e.g. '2h 15m 30s'"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    elif m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def print_progress_bar(current, total, start_time, width=30):
    """Prints a [####------] 40% | Elapsed: Xm | ETA: Ym style bar."""
    fraction = current / total if total > 0 else 0
    filled = int(width * fraction)
    bar = "#" * filled + "-" * (width - filled)
    elapsed = time.time() - start_time
    eta = (elapsed / current * (total - current)) if current > 0 else 0
    print(f"[{bar}] {fraction*100:5.1f}% | Run {current}/{total} | "
          f"Elapsed: {format_duration(elapsed)} | ETA: {format_duration(eta)}")


class HeartbeatCallback(BaseCallback):
    """
    Prints a live, in-place updating status line during model.learn() so
    the process doesn't look frozen for the ~16 minutes a single run takes.
    Updates every ~5 seconds (not every step, to avoid slowing down
    training with excessive I/O).
    """
    def __init__(self, total_timesteps, script_start_time, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.script_start_time = script_start_time
        self.run_start_time = time.time()
        self.last_print_time = 0

    def _on_step(self) -> bool:
        now = time.time()
        if now - self.last_print_time >= 5:  # update at most every 5 seconds
            self.last_print_time = now
            frac = self.num_timesteps / self.total_timesteps if self.total_timesteps else 0
            width = 20
            filled = int(width * frac)
            bar = "#" * filled + "-" * (width - filled)
            run_elapsed = now - self.run_start_time
            total_elapsed = now - self.script_start_time
            print(f"\r    training [{bar}] {frac*100:5.1f}% | "
                  f"step {self.num_timesteps}/{self.total_timesteps} | "
                  f"this run: {format_duration(run_elapsed)} | "
                  f"total elapsed: {format_duration(total_elapsed)}   ",
                  end="", flush=True)
        return True


class AdaptiveEntropyCallback(BaseCallback):
    """
    Prevents policy collapse to all-hold, which produces exactly-zero
    return/Sharpe/drawdown (no trades = portfolio never changes).
    This is the same fix your original train_rl_agent_enhanced.py used
    (EnhancedTradingCallback) — ported here since the smoke test showed
    the same hold-collapse behavior even after 15,000 training steps.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.action_counts = {0: 0, 1: 0, 2: 0}

    def _on_step(self) -> bool:
        if "actions" in self.locals:
            action = int(self.locals["actions"][0])
            self.action_counts[action] = self.action_counts.get(action, 0) + 1

        if self.n_calls % 2000 == 0 and self.n_calls > 0:
            total = sum(self.action_counts.values())
            if total > 0:
                hold_pct = self.action_counts[0] / total
                if hold_pct > 0.90 and hasattr(self.model, "ent_coef"):
                    old_ent = self.model.ent_coef
                    self.model.ent_coef = min(0.15, self.model.ent_coef * 1.5)
                    if self.verbose:
                        print(f"    [entropy boost] hold={hold_pct:.1%} -> "
                              f"ent_coef {old_ent:.4f} -> {self.model.ent_coef:.4f}")
            self.action_counts = {0: 0, 1: 0, 2: 0}
        return True

# ============================================================================
# CONFIGURATION — tune these based on your time budget
# ============================================================================
N_WINDOWS = 5
SEEDS = [42, 123, 2024]
REWARD_MODES = ["aggressive", "raw_return", "differential_sharpe"]
TRAINING_TIMESTEPS = 60000
WINDOW_SIZE = 60
INITIAL_BALANCE = 100000

RESULTS_DIR = "results/walk_forward"
os.makedirs(RESULTS_DIR, exist_ok=True)


def make_env(X_data, y_data, reward_mode, rank=0, seed=42):
    def _init():
        env = BitcoinTradingEnv(
            X_data=X_data, y_data=y_data,
            window_size=WINDOW_SIZE, initial_balance=INITIAL_BALANCE,
            mode="train" if rank == 0 else "eval",
            reward_mode=reward_mode,
        )
        env.seed(seed + rank)
        return Monitor(env)
    return _init


def evaluate_agent(model, X_test, y_test, reward_mode, seed, n_episodes=3, max_steps=None):
    """
    Run the trained agent deterministically and collect final metrics.

    BUG FIX: previously used a DummyVecEnv-wrapped eval_env here. VecEnv
    auto-resets any sub-environment the instant its episode ends (whether
    from termination OR truncation), and that reset happens INSIDE the
    same step() call that reports the episode ended. Calling
    get_final_stats() after breaking out of the loop was therefore reading
    an already-reset portfolio (portfolio_history back to [initial_balance],
    trades cleared) whenever the episode ended by reaching max_steps
    naturally — which is exactly what happens on every full-length eval
    run. This produced exactly-zero metrics regardless of what the agent
    actually did.

    Fix: use the raw (unwrapped) environment directly, not vectorized, so
    we control reset()/step() ourselves and can read get_final_stats()
    at the true end of the episode, before anything resets it.
    """
    raw_env = BitcoinTradingEnv(
        X_data=X_test, y_data=y_test,
        window_size=WINDOW_SIZE, initial_balance=INITIAL_BALANCE,
        mode="eval", reward_mode=reward_mode,
    )
    raw_env.seed(seed + 1)

    episode_metrics = []
    for _ in range(n_episodes):
        obs, _ = raw_env.reset()
        step = 0
        limit = max_steps or 100000
        terminated = False
        truncated = False

        while step < limit and not terminated and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = raw_env.step(int(action))
            step += 1

        # Read final stats NOW, before any reset happens (there is none
        # here — we control the loop directly and have not called
        # reset() again yet)
        final_stats = raw_env.get_final_stats()
        final_stats["steps_taken"] = step
        episode_metrics.append(final_stats)

    return episode_metrics


def run_single_configuration(X_full, y_full, window, reward_mode, seed, script_start_time):
    """Train + evaluate one (window, reward_mode, seed) combination."""
    train_X = X_full[window["train_start"]:window["train_end"]]
    train_y = y_full[window["train_start"]:window["train_end"]]
    test_X = X_full[window["test_start"]:window["test_end"]]
    test_y = y_full[window["test_start"]:window["test_end"]]

    set_random_seed(seed)

    train_env = DummyVecEnv([make_env(train_X, train_y, reward_mode, rank=0, seed=seed)])
    # NOTE: eval_env (vectorized) is no longer used for evaluation itself
    # (see evaluate_agent fix above) — kept here only if EvalCallback-style
    # training-time evaluation is added later. Evaluation now uses a raw
    # env constructed directly inside evaluate_agent().

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.03,   # raised from 0.01 — original project's tuned baseline used 0.02-0.05
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs={"net_arch": [256, 256]},
        verbose=0,
        seed=seed,
        device="cpu",   # small MLP policy — GPU adds overhead, not speedup, for this size
    )

    start = datetime.now()
    combined_callback = CallbackList([
        AdaptiveEntropyCallback(verbose=0),
        HeartbeatCallback(total_timesteps=TRAINING_TIMESTEPS, script_start_time=script_start_time),
    ])
    model.learn(total_timesteps=TRAINING_TIMESTEPS, callback=combined_callback,
                progress_bar=False)
    print()  # move past the in-place heartbeat line before printing the run summary
    duration = (datetime.now() - start).total_seconds()

    episode_metrics = evaluate_agent(model, test_X, test_y, reward_mode, seed, n_episodes=3,
                                      max_steps=window["test_end"] - window["test_start"] - 5)

    # Average across the 3 evaluation episodes for this single (window, mode, seed) run
    avg_metrics = {}
    for key in ["total_return", "sharpe", "sortino", "max_drawdown", "win_rate", "num_trades"]:
        values = [m[key] for m in episode_metrics if key in m]
        avg_metrics[key] = float(np.mean(values)) if values else 0.0
        avg_metrics[f"{key}_std_across_episodes"] = float(np.std(values)) if values else 0.0

    result = {
        "window": window["window"],
        "reward_mode": reward_mode,
        "seed": seed,
        "train_size": window["train_end"] - window["train_start"],
        "test_size": window["test_end"] - window["test_start"],
        "training_duration_sec": duration,
        "training_timesteps": TRAINING_TIMESTEPS,
        "reward_formula_version": REWARD_FORMULA_VERSION,
        **avg_metrics,
    }

    train_env.close()
    del model

    return result


def main():
    print("=" * 80)
    print("WALK-FORWARD VALIDATION + REWARD FUNCTION ABLATION STUDY")
    print("=" * 80)

    print("\nLoading full chronological dataset...")
    X_full = np.load("data/processed/X.npy")
    y_full = np.load("data/processed/y.npy")

    if np.isnan(X_full).any():
        X_full = np.nan_to_num(X_full, nan=0.0)
    if np.isnan(y_full).any():
        y_full = pd.Series(y_full).fillna(method="ffill").fillna(method="bfill").values

    print(f"Full dataset: X {X_full.shape}, y {y_full.shape}")

    windows = generate_walk_forward_windows(len(X_full), n_windows=N_WINDOWS)
    summarize_windows(windows, len(X_full))

    total_runs = len(windows) * len(REWARD_MODES) * len(SEEDS)
    print(f"\nTotal configurations to run: {len(windows)} windows x "
          f"{len(REWARD_MODES)} reward modes x {len(SEEDS)} seeds = {total_runs} runs")
    print(f"Estimated time at ~16 min/run (60k steps, CPU): ~{total_runs * 16 / 60:.1f} hours\n")

    # ------------------------------------------------------------------
    # RESUME SUPPORT: this run is long enough (~12 hours for the full
    # config) that a Kaggle session disconnect is a real risk. If a
    # results file already exists from a previous (interrupted) attempt,
    # load it and skip any (window, reward_mode, seed) combination that
    # already completed successfully, instead of redoing everything.
    # ------------------------------------------------------------------
    results_path = f"{RESULTS_DIR}/all_results.json"
    all_results = []
    completed = set()

    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            all_results = json.load(f)
        for r in all_results:
            if "error" not in r:
                completed.add((r["window"], r["reward_mode"], r["seed"]))
        print(f"RESUMING: found {len(completed)} already-completed runs in {results_path}")
        print("These will be skipped.\n")

    run_count = 0
    script_start_time = time.time()

    for window in windows:
        for reward_mode in REWARD_MODES:
            for seed in SEEDS:
                run_count += 1
                config_key = (window["window"], reward_mode, seed)

                print_progress_bar(run_count - 1, total_runs, script_start_time)

                if config_key in completed:
                    print(f"[{run_count}/{total_runs}] Window {window['window']} | "
                          f"Reward: {reward_mode} | Seed: {seed} -- SKIPPED (already done)")
                    continue

                print(f"[{run_count}/{total_runs}] Window {window['window']} | "
                      f"Reward: {reward_mode} | Seed: {seed} | training...")
                try:
                    result = run_single_configuration(X_full, y_full, window, reward_mode, seed,
                                                       script_start_time)
                    all_results.append(result)
                    print(f"  -> Return: {result['total_return']:+.2%} | "
                          f"Sharpe: {result['sharpe']:.2f} | "
                          f"Drawdown: {result['max_drawdown']:.2%} | "
                          f"Trades: {result['num_trades']} | "
                          f"Duration: {result['training_duration_sec']:.1f}s")
                except Exception as e:
                    print(f"  ERROR: {e}")
                    all_results.append({
                        "window": window["window"], "reward_mode": reward_mode,
                        "seed": seed, "error": str(e),
                    })

                # Save incrementally so a crash doesn't lose everything
                with open(f"{RESULTS_DIR}/all_results.json", "w") as f:
                    json.dump(all_results, f, indent=2, default=str)

    print_progress_bar(total_runs, total_runs, script_start_time)
    print(f"\n{'=' * 80}")
    print(f"COMPLETE: {len(all_results)} runs saved to {RESULTS_DIR}/all_results.json")
    print(f"Total wall-clock time this session: {format_duration(time.time() - script_start_time)}")
    print(f"{'=' * 80}")
    print("\nNext step: run evaluate_walk_forward.py to aggregate results.")


if __name__ == "__main__":
    main()
