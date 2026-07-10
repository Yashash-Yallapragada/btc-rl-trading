# updated_rl_env_v2.py
# ============================================================================
# WHY THIS EXISTS:
# Same BitcoinTradingEnv as updated_rl_env.py, but wired to
# portfolio_manager_v2.PortfolioManager instead of the original. This lets
# the SAME environment be trained/evaluated under three reward regimes
# ('aggressive', 'raw_return', 'differential_sharpe') by passing a single
# `reward_mode` argument, without touching the original files at all —
# so the original pipeline still runs untouched, and this is a clean,
# additive extension used for the walk-forward ablation study.
# ============================================================================

import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from portfolio_manager_v2 import PortfolioManager
from gym.utils import seeding


class BitcoinTradingEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(self, X_data, y_data, window_size=60, initial_balance=100000,
                 mode="train", reward_mode="raw_return", max_position_pct=0.3,
                 transaction_cost=0.0005, slippage_pct=0.0002):
        super(BitcoinTradingEnv, self).__init__()

        self.mode = mode

        if isinstance(X_data, str):
            self.X_data = np.load(X_data)
            self.y_data = np.load(y_data)
        else:
            self.X_data = X_data
            self.y_data = y_data

        if len(self.X_data.shape) == 3:
            samples, timesteps, features = self.X_data.shape
            self.features = self.X_data.reshape(samples, timesteps * features)
            self.n_features = timesteps * features
        else:
            self.features = self.X_data
            self.n_features = self.X_data.shape[1]

        if len(self.y_data.shape) > 1:
            self.prices = self.y_data.flatten()
        else:
            self.prices = self.y_data

        nan_mask = np.isnan(self.prices)
        if np.any(nan_mask):
            self.prices = pd.Series(self.prices).fillna(method="ffill").fillna(method="bfill").values
        if np.any(self.prices <= 0):
            self.prices = np.maximum(self.prices, 0.01)

        self.window_size = min(window_size, len(self.features))
        self.current_step = self.window_size
        self.max_steps = len(self.features) - 1

        self.action_space = spaces.Discrete(3)

        market_obs_size = self.window_size * self.n_features
        portfolio_obs_size = 4
        total_obs_size = market_obs_size + portfolio_obs_size

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_obs_size,), dtype=np.float32
        )

        self.portfolio = PortfolioManager(
            initial_balance=initial_balance,
            max_position_pct=max_position_pct,
            transaction_cost=transaction_cost,
            slippage_pct=slippage_pct,
            position_scaling=True,
            reward_mode=reward_mode,
        )
        self.reward_mode = reward_mode
        self.done = False
        self.portfolio_value = float(initial_balance)
        self.initial_balance = initial_balance
        self.episode_count = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.portfolio.reset()
        self.done = False
        self.episode_count += 1
        initial_price = self.prices[min(self.current_step, len(self.prices) - 1)]
        self.portfolio_value = self.portfolio._calculate_portfolio_value(initial_price)
        return self._get_observation(), {}

    def step(self, action):
        if not isinstance(action, (int, np.integer)) or action not in [0, 1, 2]:
            action = 0

        if self.current_step < len(self.prices):
            price = self.prices[self.current_step]
        else:
            price = self.prices[-1]

        if np.isnan(price) or price <= 0:
            price = self.prices[-1] if len(self.prices) > 0 else 1.0

        try:
            reward, portfolio_value = self.portfolio.update(action, price, self.current_step)
        except Exception:
            reward, portfolio_value = 0.0, self.portfolio_value

        if np.isnan(reward) or np.isinf(reward):
            reward = 0.0
        if np.isnan(portfolio_value) or np.isinf(portfolio_value) or portfolio_value <= 0:
            portfolio_value = self.portfolio_value

        self.portfolio_value = float(portfolio_value)
        self.current_step += 1

        done = False
        truncated = False
        if self.current_step >= self.max_steps:
            truncated = True
        if portfolio_value < self.initial_balance * 0.05:
            done = True
            reward -= 2.0

        obs = self._get_observation()
        portfolio_ratio = portfolio_value / self.initial_balance

        info = {
            "step": int(self.current_step),
            "max_steps": int(self.max_steps),
            "portfolio_value": float(portfolio_value),
            "portfolio_ratio": float(portfolio_ratio),
            "sharpe": float(self.portfolio.sharpe_ratio()),
            "sortino": float(self.portfolio.sortino_ratio()),
            "max_drawdown": float(self.portfolio.max_drawdown()),
            "current_price": float(price),
            "position": int(self.portfolio.position),
            "num_trades": int(len([t for t in self.portfolio.trades if "profit" in t])),
            "reward_mode": self.reward_mode,
        }

        return obs, float(reward), done, truncated, info

    def _get_observation(self):
        start_idx = max(0, self.current_step - self.window_size)
        end_idx = self.current_step
        if end_idx > len(self.features):
            end_idx = len(self.features)
            start_idx = max(0, end_idx - self.window_size)

        window_data = self.features[start_idx:end_idx]
        if len(window_data) < self.window_size:
            padding = np.zeros((self.window_size - len(window_data), self.n_features))
            window_data = np.vstack([padding, window_data])

        if np.any(np.isnan(window_data)) or np.any(np.isinf(window_data)):
            window_data = np.nan_to_num(window_data, nan=0.0, posinf=0.0, neginf=0.0)

        market_obs = window_data.flatten().astype(np.float32)

        current_price = self.prices[min(self.current_step, len(self.prices) - 1)]
        position_pnl = 0.0
        if self.portfolio.position != 0 and self.portfolio.entry_price is not None:
            if self.portfolio.position == 1:
                position_pnl = (current_price - self.portfolio.entry_price) / self.portfolio.entry_price
            else:
                position_pnl = (self.portfolio.entry_price - current_price) / self.portfolio.entry_price

        portfolio_obs = np.array([
            self.portfolio.cash / self.initial_balance,
            float(self.portfolio.position),
            self.portfolio_value / self.initial_balance,
            position_pnl,
        ], dtype=np.float32)

        full_obs = np.concatenate([market_obs, portfolio_obs])
        if np.any(np.isnan(full_obs)) or np.any(np.isinf(full_obs)):
            full_obs = np.nan_to_num(full_obs, nan=0.0, posinf=1.0, neginf=-1.0)
        return full_obs

    def render(self, mode="human"):
        price = self.prices[min(self.current_step, len(self.prices) - 1)]
        metrics = self.portfolio.get_portfolio_metrics(price)
        print(f"[{self.reward_mode}] Step {self.current_step}/{self.max_steps} | "
              f"Portfolio: ${metrics['final_value']:.2f} | Return: {metrics['total_return']:.1%} | "
              f"Sharpe: {metrics['sharpe']:.2f}")

    def get_final_stats(self):
        current_price = self.prices[min(self.current_step, len(self.prices) - 1)]
        return self.portfolio.get_portfolio_metrics(current_price)

    def close(self):
        pass
