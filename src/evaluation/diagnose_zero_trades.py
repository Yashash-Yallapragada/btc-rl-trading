# diagnose_zero_trades.py
# ============================================================================
# The smoke test shows EXACTLY zero return/Sharpe/drawdown even after
# 15,000 steps with a raised entropy coefficient. This script instruments
# both training and evaluation to find the real cause, rather than guessing
# further. It checks three hypotheses:
#
#   1. Is the agent actually exploring during training (action distribution)?
#   2. What does the raw price data actually look like step-to-step —
#      is the MinMax-scaled "price" moving enough per hourly step to
#      produce a meaningful reward signal at all?
#   3. Does the deterministic eval policy ever pick buy/sell, even once?
# ============================================================================

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from updated_rl_env_v2 import BitcoinTradingEnv

WINDOW_SIZE = 60


class ActionTrackingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.action_log = []

    def _on_step(self) -> bool:
        if "actions" in self.locals:
            self.action_log.append(int(self.locals["actions"][0]))
        if self.n_calls % 3000 == 0 and self.n_calls > 0:
            recent = self.action_log[-3000:]
            counts = {a: recent.count(a) for a in [0, 1, 2]}
            total = sum(counts.values())
            print(f"    Step {self.n_calls}: Hold={counts[0]/total:.1%} "
                  f"Buy={counts[1]/total:.1%} Sell={counts[2]/total:.1%} "
                  f"| ent_coef={self.model.ent_coef if hasattr(self.model, 'ent_coef') else 'n/a'}")
        return True


def make_env(X_data, y_data, reward_mode, rank=0, seed=42):
    def _init():
        env = BitcoinTradingEnv(
            X_data=X_data, y_data=y_data,
            window_size=WINDOW_SIZE, initial_balance=100000,
            mode="train" if rank == 0 else "eval",
            reward_mode=reward_mode,
        )
        env.seed(seed + rank)
        return Monitor(env)
    return _init


def inspect_price_scale(y_data, n_samples=20):
    print("\n" + "=" * 80)
    print("RAW PRICE SCALE CHECK (first 20 steps of test data)")
    print("=" * 80)
    print(f"y_data range: {y_data.min():.6f} to {y_data.max():.6f}")
    print(f"y_data mean: {y_data.mean():.6f}, std: {y_data.std():.6f}\n")

    for i in range(min(n_samples, len(y_data) - 1)):
        p1, p2 = y_data[i], y_data[i + 1]
        pct_change = (p2 - p1) / p1 if p1 != 0 else float("nan")
        print(f"  step {i}: price={p1:.6f} -> {p2:.6f} | change={pct_change:+.6%}")

    # Overall step-to-step change statistics
    changes = np.diff(y_data) / (y_data[:-1] + 1e-10)
    print(f"\nStep-to-step %% change stats across full series:")
    print(f"  mean: {np.mean(changes):+.6%}")
    print(f"  std:  {np.std(changes):.6%}")
    print(f"  min:  {np.min(changes):+.6%}")
    print(f"  max:  {np.max(changes):+.6%}")


def main():
    print("Loading data...")
    X_full = np.load("data/processed/X.npy")
    y_full = np.load("data/processed/y.npy")

    # Use a small slice for a fast diagnostic
    train_X, train_y = X_full[:5000], y_full[:5000]
    test_X, test_y = X_full[5000:6000], y_full[5000:6000]

    inspect_price_scale(test_y)

    print("\n" + "=" * 80)
    print("TRAINING WITH ACTION TRACKING (raw_return reward mode)")
    print("=" * 80)

    train_env = DummyVecEnv([make_env(train_X, train_y, "raw_return", rank=0)])
    eval_env = DummyVecEnv([make_env(test_X, test_y, "raw_return", rank=1)])

    model = PPO(
        policy="MlpPolicy", env=train_env,
        learning_rate=3e-4, n_steps=1024, batch_size=64, n_epochs=10,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.03,
        vf_coef=0.5, max_grad_norm=0.5,
        policy_kwargs={"net_arch": [256, 256]}, verbose=0, seed=42,
    )

    callback = ActionTrackingCallback(verbose=1)
    model.learn(total_timesteps=15000, callback=callback, progress_bar=False)

    print("\n" + "=" * 80)
    print("DETERMINISTIC EVAL — action-by-action for first 50 steps")
    print("=" * 80)

    obs = eval_env.reset()
    action_names = {0: "HOLD", 1: "BUY", 2: "SELL"}
    eval_actions = []
    for step in range(50):
        action, _ = model.predict(obs, deterministic=True)
        eval_actions.append(int(action[0]))
        obs, reward, done, info = eval_env.step(action)
        if step < 20:
            print(f"  step {step}: action={action_names[int(action[0])]:4s} "
                  f"reward={reward[0]:+.6f} portfolio=${info[0]['portfolio_value']:.2f}")
        if done[0]:
            print(f"  [episode ended early at step {step}]")
            break

    counts = {a: eval_actions.count(a) for a in [0, 1, 2]}
    total = len(eval_actions)
    print(f"\nEval action distribution over {total} steps: "
          f"Hold={counts[0]/total:.1%} Buy={counts[1]/total:.1%} Sell={counts[2]/total:.1%}")

    print("\n" + "=" * 80)
    print("STOCHASTIC EVAL (deterministic=False) — sanity check")
    print("=" * 80)
    obs = eval_env.reset()
    stoch_actions = []
    for step in range(50):
        action, _ = model.predict(obs, deterministic=False)
        stoch_actions.append(int(action[0]))
        obs, reward, done, info = eval_env.step(action)
        if done[0]:
            break
    counts_s = {a: stoch_actions.count(a) for a in [0, 1, 2]}
    total_s = len(stoch_actions)
    print(f"Stochastic action distribution: "
          f"Hold={counts_s[0]/total_s:.1%} Buy={counts_s[1]/total_s:.1%} Sell={counts_s[2]/total_s:.1%}")


if __name__ == "__main__":
    main()