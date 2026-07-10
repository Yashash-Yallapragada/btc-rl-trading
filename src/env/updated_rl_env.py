# updated_rl_env_optimized.py - OPTIMIZED VERSION
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from env_helper_portfolio_original import PortfolioManager
from gym.utils import seeding

class BitcoinTradingEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(self, X_data, y_data, window_size=60, initial_balance=100000, mode="train"):
        super(BitcoinTradingEnv, self).__init__()

        self.mode = mode

        # Handle different input formats
        if isinstance(X_data, str):
            self.X_data = np.load(X_data)
            self.y_data = np.load(y_data)
        else:
            self.X_data = X_data
            self.y_data = y_data

        # Handle different data shapes
        if len(self.X_data.shape) == 3:
            # 3D data: (samples, timesteps, features) - flatten timesteps
            samples, timesteps, features = self.X_data.shape
            self.features = self.X_data.reshape(samples, timesteps * features)
            self.n_features = timesteps * features
            print(f"[3D DATA FLATTENED] Original: {self.X_data.shape} -> Flattened: {self.features.shape}")
        else:
            # 2D data: (samples, features)
            self.features = self.X_data
            self.n_features = self.X_data.shape[1]
            print(f"[2D DATA] Samples: {self.X_data.shape[0]}, Features: {self.X_data.shape[1]}")

        # Handle prices - validate for NaN values
        if len(self.y_data.shape) > 1:
            self.prices = self.y_data.flatten()
        else:
            self.prices = self.y_data

        # Check for NaN prices and handle them
        nan_mask = np.isnan(self.prices)
        if np.any(nan_mask):
            print(f"[WARNING] Found {np.sum(nan_mask)} NaN prices, forward-filling...")
            self.prices = pd.Series(self.prices).fillna(method='ffill').fillna(method='bfill').values
        
        # Validate price range
        if np.any(self.prices <= 0):
            print(f"[WARNING] Found non-positive prices, setting minimum to 0.01")
            self.prices = np.maximum(self.prices, 0.01)

        # Find better starting point with price variation
        def find_varied_start_index(prices, min_variation=0.02, window=50):
            for i in range(len(prices) - window):
                window_data = prices[i:i+window]
                if np.std(window_data) > min_variation and len(np.unique(window_data)) > window * 0.7:
                    return i
            return window_size

        varied_start = find_varied_start_index(self.prices)
        print(f"[INFO] Starting at index {varied_start} for better price variation")

        self.window_size = min(window_size, len(self.features))
        self.current_step = max(self.window_size, varied_start)
        self.max_steps = len(self.features) - 1

        # Action space: 0 = hold, 1 = buy, 2 = sell
        self.action_space = spaces.Discrete(3)
        
        # Observation space for window-based features + portfolio state
        market_obs_size = self.window_size * self.n_features
        portfolio_obs_size = 4  # cash_ratio, position, portfolio_ratio, position_pnl
        total_obs_size = market_obs_size + portfolio_obs_size
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_obs_size,),  # Flattened observation
            dtype=np.float32
        )

        # Optimized portfolio manager
        self.portfolio = PortfolioManager(
            initial_balance=initial_balance,
            max_position_pct=0.8,  # More aggressive
            transaction_cost=0.0005,  # Lower costs
            position_scaling=True
        )
        self.done = False
        
        # Initialize portfolio_value
        self.portfolio_value = float(initial_balance)
        self.initial_balance = initial_balance
        
        # Episode tracking
        self.episode_count = 0

        print(f"[INIT] Window size: {self.window_size}, n_features: {self.n_features}")
        print(f"[INIT] Observation shape: {self.observation_space.shape}")
        print(f"[INIT] Price range: ${np.min(self.prices):.2f} - ${np.max(self.prices):.2f}")

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Start from varied region
        varied_start = max(self.window_size, 
                          min(450, len(self.features) - 100))  # Safe fallback
        self.current_step = varied_start
        
        self.portfolio.reset()
        self.done = False
        self.episode_count += 1
        
        # Initialize portfolio_value properly
        initial_price = self.prices[min(self.current_step, len(self.prices)-1)]
        self.portfolio_value = self.portfolio._calculate_portfolio_value(initial_price)
        
        return self._get_observation(), {}

    def step(self, action):
        # Validate action
        if not isinstance(action, (int, np.integer)) or action not in [0, 1, 2]:
            action = 0
        
        # Get current price with bounds checking
        if self.current_step < len(self.prices):
            price = self.prices[self.current_step]
        else:
            price = self.prices[-1]
        
        # Validate price
        if np.isnan(price) or price <= 0:
            price = self.prices[-1] if len(self.prices) > 0 else 1.0

        # Update portfolio with error handling
        try:
            reward, portfolio_value = self.portfolio.update(action, price, self.current_step)
        except Exception as e:
            print(f"[ERROR] Portfolio update failed: {e}")
            reward, portfolio_value = 0.0, self.portfolio_value

        # Enhanced reward shaping for better learning
        if self.current_step > 0:
            prev_price = self.prices[self.current_step - 1]
            price_change_pct = (price - prev_price) / prev_price if prev_price > 0 else 0.0

            # Reward for correct directional trades
            if action == 1 and price_change_pct > 0.001:  # Buy before price up
                reward += 0.1
            elif action == 2 and price_change_pct < -0.001:  # Sell before price down
                reward += 0.1
            
            # Small penalty for wrong direction
            elif action == 1 and price_change_pct < -0.001:  # Buy before price down
                reward -= 0.05
            elif action == 2 and price_change_pct > 0.001:  # Sell before price up
                reward -= 0.05

        # Momentum reward for holding profitable positions
        if hasattr(self, 'last_portfolio_value'):
            value_change = portfolio_value - self.last_portfolio_value
            if value_change > 0 and self.portfolio.position != 0:
                reward += 0.02  # Small bonus for profitable positions

        # Store for next step
        self.last_portfolio_value = portfolio_value

        # Validate outputs
        if np.isnan(reward) or np.isinf(reward):
            reward = 0.0
            
        if np.isnan(portfolio_value) or np.isinf(portfolio_value) or portfolio_value <= 0:
            portfolio_value = self.portfolio_value

        # Update portfolio value
        self.portfolio_value = float(portfolio_value)

        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        done = False
        truncated = False
        
        if self.current_step >= self.max_steps:
            truncated = True
            
        # Only stop for truly catastrophic losses (95% loss)
        if portfolio_value < self.initial_balance * 0.05:
            done = True
            reward -= 5.0

        # Get next observation
        obs = self._get_observation()

        # Create info dict
        portfolio_ratio = portfolio_value / self.initial_balance
        info = {
            "step": int(self.current_step),
            "max_steps": int(self.max_steps),
            "portfolio_value": float(portfolio_value),
            "portfolio_ratio": float(portfolio_ratio),
            "initial_balance": float(self.initial_balance),
            "sharpe": float(self.portfolio.sharpe_ratio()),
            "max_drawdown": float(self.portfolio.max_drawdown()),
            "current_price": float(price),
            "position": int(self.portfolio.position),
            "total_reward": float(self.portfolio.total_reward),
            "num_trades": int(len([t for t in self.portfolio.trades if 'profit' in t])),
            "episode": {
                "r": float(self.portfolio.total_reward),
                "l": int(self.current_step)
            }
        }
        
        return obs, float(reward), done, truncated, info

    def _get_observation(self):
        """Get flattened observation including market data and portfolio state"""
        # Get market data window
        start_idx = max(0, self.current_step - self.window_size)
        end_idx = self.current_step

        if end_idx > len(self.features):
            end_idx = len(self.features)
            start_idx = max(0, end_idx - self.window_size)

        window_data = self.features[start_idx:end_idx]

        # Pad if necessary
        if len(window_data) < self.window_size:
            padding = np.zeros((self.window_size - len(window_data), self.n_features))
            window_data = np.vstack([padding, window_data])

        # Validate and flatten market data
        if np.any(np.isnan(window_data)) or np.any(np.isinf(window_data)):
            window_data = np.nan_to_num(window_data, nan=0.0, posinf=0.0, neginf=0.0)

        market_obs = window_data.flatten().astype(np.float32)

        # Portfolio state features
        current_price = self.prices[min(self.current_step, len(self.prices)-1)]
        
        # Calculate position P&L if we have a position
        position_pnl = 0.0
        if self.portfolio.position != 0 and self.portfolio.entry_price is not None:
            if self.portfolio.position == 1:  # Long
                position_pnl = (current_price - self.portfolio.entry_price) / self.portfolio.entry_price
            else:  # Short
                position_pnl = (self.portfolio.entry_price - current_price) / self.portfolio.entry_price

        portfolio_obs = np.array([
            self.portfolio.cash / self.initial_balance,  # Cash ratio
            float(self.portfolio.position),              # Position (-1, 0, 1)
            self.portfolio_value / self.initial_balance, # Portfolio ratio
            position_pnl                                 # Current position P&L
        ], dtype=np.float32)

        # Combine observations
        full_obs = np.concatenate([market_obs, portfolio_obs])
        
        # Final validation
        if np.any(np.isnan(full_obs)) or np.any(np.isinf(full_obs)):
            full_obs = np.nan_to_num(full_obs, nan=0.0, posinf=1.0, neginf=-1.0)

        return full_obs

    def render(self, mode="human"):
        if mode == "human":
            price = self.prices[min(self.current_step, len(self.prices)-1)]
            metrics = self.portfolio.get_portfolio_metrics(price)
            pct_complete = (self.current_step / self.max_steps) * 100
            
            print(f"Episode {self.episode_count} | Step: {self.current_step:4}/{self.max_steps} ({pct_complete:5.1f}%) | "
                  f"Price: ${price:7.2f} | Portfolio: ${metrics['final_value']:8.2f} | "
                  f"Return: {metrics['total_return']:6.1%} | Position: {self.portfolio.position:2} | "
                  f"Trades: {metrics['num_trades']:2}")

    def get_final_stats(self):
        """Get comprehensive final statistics"""
        current_price = self.prices[min(self.current_step, len(self.prices)-1)]
        return self.portfolio.get_portfolio_metrics(current_price)

    def close(self):
        """Clean up the environment"""
        pass