# updated_rl_env_with_lstm.py - FIXED VERSION

import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from env_helper_portfolio import PortfolioManager
from gym.utils import seeding
import tensorflow as tf

class BitcoinTradingEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(self, X_data, y_data, lstm_model=None, scaler=None, window_size=60, 
                 initial_balance=100000, mode="train", prediction_horizon=1):
        super(BitcoinTradingEnv, self).__init__()

        self.mode = mode
        self.lstm_model = lstm_model
        self.scaler = scaler
        self.prediction_horizon = prediction_horizon

        # Handle different input formats
        if isinstance(X_data, str):
            self.X_data = np.load(X_data)
            self.y_data = np.load(y_data)
        else:
            self.X_data = X_data
            self.y_data = y_data

        # Handle 3D arrays (LSTM sequences) vs 2D arrays
        if len(self.X_data.shape) == 3:
            self.features = self.X_data
            self.n_features = self.X_data.shape[2]
            self.sequence_length = self.X_data.shape[1]
            self.use_sequences = True
            print(f"[3D DATA] Samples: {self.X_data.shape[0]}, Timesteps: {self.X_data.shape[1]}, Features: {self.X_data.shape[2]}")
        else:
            self.features = self.X_data
            self.n_features = self.X_data.shape[1]
            self.sequence_length = window_size
            self.use_sequences = False
            print(f"[2D DATA] Samples: {self.X_data.shape[0]}, Features: {self.X_data.shape[1]}")

        # Handle prices with enhanced validation
        if len(self.y_data.shape) > 1:
            self.prices = self.y_data.flatten()
        else:
            self.prices = self.y_data

        # Enhanced price validation and cleaning
        nan_mask = np.isnan(self.prices)
        if np.any(nan_mask):
            print(f"[WARNING] Found {np.sum(nan_mask)} NaN prices, forward-filling...")
            self.prices = pd.Series(self.prices).fillna(method='ffill').fillna(method='bfill').values
        
        # Ensure all prices are positive and reasonable
        if np.any(self.prices <= 0):
            print(f"[WARNING] Found {np.sum(self.prices <= 0)} non-positive prices, fixing...")
            self.prices = np.maximum(self.prices, 0.01)

        # ADD THIS NEW SECTION - Find a starting point with price variation
        def find_varied_start_index(prices, min_variation=0.01):
            for i in range(len(prices) - 100):
                window = prices[i:i+50]
                if np.std(window) > min_variation:
                    return i
            return 0

        varied_start = find_varied_start_index(self.prices)
        print(f"[INFO] Found varied data starting at index {varied_start}")

        self.window_size = min(window_size, len(self.features))
        self.current_step = max(self.window_size, varied_start)  # Use varied start instead of just window_size
        self.max_steps = len(self.features) - 1


        # Action space: 0 = hold, 1 = buy, 2 = sell
        self.action_space = spaces.Discrete(3)
        
        # FIXED: Calculate observation space dimensions correctly
        if self.use_sequences:
            # For 3D data, flatten the sequence dimension
            base_features = self.sequence_length * self.n_features
        else:
            # For 2D data, use single timestep
            base_features = self.n_features
        
        # Portfolio state features
        portfolio_features = 3  # cash_ratio, position, portfolio_ratio
        
        # LSTM prediction features
        lstm_features = 3 if self.lstm_model is not None else 0
        
        # Total observation size (1D array)
        total_obs_size = base_features + portfolio_features + lstm_features
        
        # FIXED: Use 1D observation space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_obs_size,),  # 1D shape!
            dtype=np.float32
        )

        print(f"[OBS SPACE] 1D Shape: ({total_obs_size},) = Base:{base_features} + Portfolio:{portfolio_features} + LSTM:{lstm_features}")

        # Initialize portfolio with reasonable settings
        self.portfolio = PortfolioManager(
            initial_balance=initial_balance,
            max_position_pct=0.2,
            transaction_cost=0.001,
            position_scaling=False
        )
        
        # Add environment reference for LSTM logging
        self.portfolio.current_env_ref = self

        self.done = False
        self.initial_balance = initial_balance
        self.portfolio_value = float(initial_balance)
        self.episode_count = 0
        
        print(f"[INIT] Window size: {self.window_size}, n_features: {self.n_features}")
        print(f"[INIT] Price range: ${np.min(self.prices):.2f} - ${np.max(self.prices):.2f}")

    def _get_lstm_predictions(self, current_step):
        """Generate LSTM predictions with proper sequence handling"""
        if self.lstm_model is None:
            return np.array([0.0, 0.0, 0.0])
        
        try:
            # Get proper sequence for LSTM prediction
            end_idx = min(current_step + 1, len(self.features))
            start_idx = max(0, end_idx - self.sequence_length)
            
            if self.use_sequences:
                # For 3D data, get the sequence at current step
                if current_step < len(self.features):
                    sequence = self.features[current_step]
                else:
                    sequence = self.features[-1]
            else:
                # For 2D data, create sequence from window
                sequence = self.features[start_idx:end_idx]
                
                # Pad if necessary
                if len(sequence) < self.sequence_length:
                    padding = np.zeros((self.sequence_length - len(sequence), self.n_features))
                    sequence = np.vstack([padding, sequence])
                elif len(sequence) > self.sequence_length:
                    sequence = sequence[-self.sequence_length:]
            
            # Reshape for LSTM input
            lstm_input = sequence.reshape(1, self.sequence_length, self.n_features)
            
            # Apply scaling if available
            if self.scaler is not None:
                original_shape = lstm_input.shape
                lstm_input_scaled = lstm_input.reshape(-1, lstm_input.shape[-1])
                lstm_input_scaled = self.scaler.transform(lstm_input_scaled)
                lstm_input = lstm_input_scaled.reshape(original_shape)
            
            # Get prediction
            raw_prediction = self.lstm_model.predict(lstm_input, verbose=0)[0, 0]
            
            # Get current price for comparison
            current_price = self.prices[min(current_step, len(self.prices)-1)]
            
            # Validate prediction
            if np.isnan(raw_prediction) or np.isinf(raw_prediction) or raw_prediction <= 0:
                raw_prediction = current_price
            
            # Calculate price change percentage
            price_change_pct = (raw_prediction - current_price) / current_price if current_price > 0 else 0.0
            price_change_pct = np.clip(price_change_pct, -0.5, 0.5)  # Cap at ±50%
            
            # Simple confidence score based on magnitude of prediction
            confidence_score = min(abs(price_change_pct) * 2, 1.0)
            
            return np.array([raw_prediction, price_change_pct, confidence_score])
            
        except Exception as e:
            print(f"[LSTM ERROR] Prediction failed: {e}")
            return np.array([0.0, 0.0, 0.0])

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.portfolio.reset()
        self.done = False
        self.episode_count += 1
        
        initial_price = self.prices[min(self.current_step, len(self.prices)-1)]
        self.portfolio_value = self.portfolio._calculate_portfolio_value(initial_price)
        
        return self._get_observation(), {}

    def step(self, action):
        # Validate action
        if not isinstance(action, (int, np.integer)) or action not in [0, 1, 2]:
            print(f"[WARNING] Invalid action {action}, defaulting to 0 (hold)")
            action = 0
        
        # Get current price
        if self.current_step < len(self.prices):
            price = self.prices[self.current_step]
        else:
            price = self.prices[-1]
        
        # DEBUG: Print every step to see price progression
        print(f"Step {self.current_step}: Price=${price:.4f}")
        
        # Validate price
        if np.isnan(price) or price <= 0:
            print(f"[WARNING] Invalid price at step {self.current_step}: {price}")
            if self.current_step > 0:
                price = self.prices[self.current_step - 1]
            else:
                price = 100.0

        # Update portfolio
        try:
            reward, portfolio_value = self.portfolio.update(action, price, self.current_step)
        except Exception as e:
            print(f"[ERROR] Portfolio update failed: {e}")
            reward, portfolio_value = -0.1, self.portfolio_value

        # SIMPLIFIED reward modifications - only essential ones
        if np.isnan(portfolio_value) or np.isinf(portfolio_value) or portfolio_value <= 0:
            print(f"[CRITICAL] Invalid portfolio value: {portfolio_value}")
            portfolio_value = max(self.portfolio.cash, 1.0)
            reward = -1.0
        
        # Small transaction cost penalty (keep trading realistic)
        if action != 0:  # Buy or sell
            reward -= 0.01
        
        # Emergency stop for catastrophic losses only
        if portfolio_value < self.initial_balance * 0.05:  # Lost 95% or more
            reward = -5.0
        
        # Clip reward to reasonable bounds
        reward = np.clip(reward, -5.0, 5.0)

        # Update portfolio value
        self.portfolio_value = float(portfolio_value)

        # Move to next step
        self.current_step += 1
        
        # Check termination conditions
        done = False
        truncated = False
        
        # Normal episode completion
        if self.current_step >= self.max_steps:
            truncated = True
            
        # Only stop for truly catastrophic situations
        if portfolio_value <= self.initial_balance * 0.05:  # Lost 95%
            done = True
            print(f"[EMERGENCY STOP] Portfolio critically low: ${portfolio_value:.2f}")

        # Get next observation
        obs = self._get_observation()

        # Create info dictionary
        portfolio_ratio = portfolio_value / self.initial_balance
        info = {
            "step": int(self.current_step),
            "max_steps": int(self.max_steps),
            "portfolio_value": float(portfolio_value),
            "portfolio_ratio": float(portfolio_ratio),
            "current_price": float(price),
            "position": int(self.portfolio.position),
            "cash": float(self.portfolio.cash),
            "episode": {
                "r": float(reward),
                "l": int(self.current_step)
            }
        }
        
        return obs, float(reward), done, truncated, info
    def _get_observation(self):
        """FIXED: Return consistent 1D observation"""
        current_idx = min(self.current_step, len(self.features)-1)
        
        # Get market features
        if self.use_sequences:
            # For 3D data, flatten the current sequence
            market_features = self.features[current_idx].flatten()
        else:
            # For 2D data, use current row
            market_features = self.features[current_idx]
        
        # Portfolio state features
        portfolio_features = np.array([
            self.portfolio.cash / self.initial_balance,           # Cash ratio
            float(self.portfolio.position),                       # Position (-1, 0, 1)
            self.portfolio_value / self.initial_balance          # Portfolio ratio
        ])
        
        # LSTM prediction features
        if self.lstm_model is not None:
            lstm_pred = self._get_lstm_predictions(current_idx)
            obs = np.concatenate([market_features, portfolio_features, lstm_pred])
        else:
            obs = np.concatenate([market_features, portfolio_features])
        
        # Validate observation
        if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
            print(f"[WARNING] Invalid observation data, cleaning...")
            obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return obs.astype(np.float32)

    def render(self, mode="human"):
        if mode == "human":
            price = self.prices[min(self.current_step, len(self.prices)-1)]
            portfolio_ratio = self.portfolio_value / self.initial_balance
            pct_complete = (self.current_step / self.max_steps) * 100
            
            print(f"Episode {self.episode_count} | Step: {self.current_step:4}/{self.max_steps} "
                  f"({pct_complete:5.1f}%) | Price: ${price:7.2f} | "
                  f"Portfolio: ${self.portfolio_value:8.2f} ({portfolio_ratio:.3f}x) | "
                  f"Position: {self.portfolio.position:2}")

    def get_final_stats(self):
        """Get final episode statistics"""
        current_price = self.prices[min(self.current_step, len(self.prices)-1)]
        base_stats = self.portfolio.get_portfolio_metrics(current_price)
        
        base_stats.update({
            "portfolio_ratio": float(base_stats['final_value'] / self.initial_balance),
            "episode_length": int(self.current_step)
        })
        
        return base_stats

    def close(self):
        """Clean up the environment"""
        pass