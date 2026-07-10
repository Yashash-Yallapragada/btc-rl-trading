# train_rl_agent.py - WITH LSTM INTEGRATION ADDED

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
import os
import pandas as pd
from datetime import datetime
import pickle
import tensorflow as tf

# Import custom environment and seed
from updated_rl_env_with_lstm import BitcoinTradingEnv  # Updated import
from environment_setup import SEED

# Set seed
set_random_seed(SEED)

print("🚀 Starting RL Agent Training Setup")
print("=" * 60)

# ===========================
# 1. LOAD LSTM MODEL (NEW)
# ===========================
print("🧠 Loading LSTM model...")

lstm_model = None
scaler = None

try:
    lstm_model = tf.keras.models.load_model(
        'models/lstm_price_predictor.h5',
        custom_objects={'mse': 'mean_squared_error'}
    )
    print("✅ LSTM model loaded successfully")
except Exception as e:
    print(f"❌ LSTM model loading failed: {e}")
    print("🔄 Trying to rebuild model...")
    
    try:
        # Fallback: rebuild model and load weights only
        from lstm_model import build_lstm_model
        X_sample = np.load('data/processed/X_train.npy')
        input_shape = (X_sample.shape[1], X_sample.shape[2])
        lstm_model = build_lstm_model(input_shape)
        lstm_model.load_weights('models/lstm_best_weights.h5')
        print("✅ LSTM weights loaded successfully")
    except:
        print("⚠️ No LSTM weights found, proceeding without LSTM")
        lstm_model = None

try:
    import joblib
    scaler = joblib.load('models/scaler.pkl')
    print("✅ Scaler loaded successfully")
except:
    print("⚠️ No scaler found, proceeding without scaling")

# ===========================
# 2. LOAD DATA
# ===========================
print("📊 Loading training data...")

X_train = np.load('data/processed/X_train.npy')
y_train = np.load('data/processed/y_train_clean.npy')  # Use cleaned version)
X_val = np.load('data/processed/X_val.npy')
y_val = np.load('data/processed/y_val.npy')
x_test = np.load('data/processed/X_test.npy')
y_test = np.load('data/processed/y_test.npy')

print("Checking for NaN values in training data...")
print(f"X_train NaN count: {np.isnan(X_train).sum()}")
print(f"y_train NaN count: {np.isnan(y_train).sum()}")
print(f"X_val NaN count: {np.isnan(X_val).sum()}")  
print(f"y_val NaN count: {np.isnan(y_val).sum()}")

# Remove or replace NaN values if found
if np.isnan(X_train).any():
    print("Replacing NaN values in X_train with 0")
    X_train = np.nan_to_num(X_train, nan=0.0)
    
if np.isnan(y_train).any():
    print("Forward filling NaN values in y_train")
    y_train = pd.Series(y_train).fillna(method='ffill').fillna(method='bfill').values

if np.isnan(X_val).any():
    print("Replacing NaN values in X_val with 0")
    X_val = np.nan_to_num(X_val, nan=0.0)
    
if np.isnan(y_val).any():
    print("Forward filling NaN values in y_val")
    y_val = pd.Series(y_val).fillna(method='ffill').fillna(method='bfill').values

# Fix shape if transposed (applies to both train & val)
for arr_name, arr in [('X_train', X_train), ('X_val', X_val)]:
    if arr.shape[0] < arr.shape[1]: 
        print(f"[SHAPE FIX] Transposing {arr_name} from {arr.shape} → {arr.T.shape}")
        arr = arr.T
    if arr_name == 'X_train':
        X_train = arr
    else:
        X_val = arr
    print(f"[SHAPE CHECK] {arr_name} shape: {arr.shape}")

print(f"✅ Training data loaded: X_train {X_train.shape}, y_train {y_train.shape}")
print(f"✅ Validation data loaded: X_val {X_val.shape}, y_val {y_val.shape}")
print(f"📈 Training Price Range: ${y_train.min():.2f} - ${y_train.max():.2f}")
print(f"📈 Validation Price Range: ${y_val.min():.2f} - ${y_val.max():.2f}")

# ===========================
# 3. CREATE ENVIRONMENTS (UPDATED FOR LSTM)
# ===========================
print("\n🏗️ Creating training environments...")

os.makedirs('models', exist_ok=True)
os.makedirs('logs', exist_ok=True)
os.makedirs('tensorboard_logs', exist_ok=True)

def make_env(X_data, y_data, lstm_model=None, scaler=None, rank=0, seed=SEED):
    """
    Factory to create environments with LSTM integration.
    Automatically fixes transposes and sets window_size correctly.
    """
    # Don't transpose 3D data - it's already in correct format (samples, timesteps, features)
    if X_data.ndim == 2:
        # For 2D data, ensure shape is (timesteps, features)
        if X_data.shape[0] < X_data.shape[1]:
            print(f"[SHAPE FIX] Transposing 2D data from {X_data.shape} to {X_data.T.shape}")
            X_data = X_data.T
        timesteps, n_features = X_data.shape
        window_size = min(60, timesteps)  # Use reasonable window size, not all timesteps
        
    elif X_data.ndim == 3:
        # For 3D data: (samples, timesteps, features) - DO NOT transpose
        samples, timesteps, n_features = X_data.shape
        window_size = timesteps  # Use the sequence length from the data
        print(f"[3D ENV] Samples: {samples}, Timesteps: {timesteps}, Features: {n_features}")
        
    else:
        raise ValueError(f"Unexpected X_data shape: {X_data.shape}")

    print(f"[ENV INIT] Using window_size: {window_size}, n_features: {n_features}")
    print(f"[LSTM STATUS] Model available: {lstm_model is not None}")

    def _init():
        env = BitcoinTradingEnv(
            X_data=X_data,
            y_data=y_data,
            lstm_model=lstm_model,  # Added LSTM
            scaler=scaler,          # Added scaler
            window_size=window_size,
            initial_balance=100000
        )
        env.seed(seed + rank)
        env = Monitor(env)
        return env

    return _init

train_env = DummyVecEnv([make_env(X_train, y_train, lstm_model, scaler, rank=0)])
eval_env = DummyVecEnv([make_env(x_test, y_test, lstm_model, scaler, rank=1)])

print("✅ Environments created successfully!")
print(f"🧠 LSTM Integration: {'Enabled' if lstm_model else 'Disabled'}")

# ===========================
# 4. PPO CONFIGURATION (UNCHANGED)
# ===========================
print("\n⚙️ Configuring PPO Agent...")

from torch import nn

ppo_config = {
    'policy': 'MlpPolicy',
    'env': train_env,
    'learning_rate': 3e-4,
    'n_steps': 2048,
    'batch_size': 64,
    'n_epochs': 10,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_range': 0.2,
    'clip_range_vf': None,
    'ent_coef': 0.01,
    'vf_coef': 0.5,
    'max_grad_norm': 0.5,
    'use_sde': False,
    'sde_sample_freq': -1,
    'target_kl': None,
    'tensorboard_log': 'tensorboard_logs/',
    'policy_kwargs': {
        'net_arch': [256, 256],
        'activation_fn': nn.ReLU
    },
    'verbose': 1,
    'seed': SEED
}

print("📋 PPO Configuration:")
for key, value in ppo_config.items():
    if key != 'env':
        print(f"   {key}: {value}")

# ===========================
# 5. INITIALIZE PPO AGENT (UNCHANGED)
# ===========================
print("\n🤖 Initializing PPO Agent...")

model = PPO(**ppo_config)

from stable_baselines3.common.callbacks import BaseCallback

class ActionLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(ActionLoggingCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        if self.n_calls % 1000 == 0:
            print(f"[TRAIN DEBUG] Step: {self.n_calls}, Num Timesteps: {self.num_timesteps}")
        return True

print("✅ PPO Agent initialized successfully!")
print(f"📊 Policy architecture: {model.policy}")

# ===========================
# 6. CALLBACKS (UNCHANGED)
# ===========================
print("\n📊 Setting up training callbacks...")

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path='models/',
    log_path='logs/',
    eval_freq=5000,
    deterministic=True,
    render=False,
    n_eval_episodes=5,
    verbose=1
)

checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path='models/',
    name_prefix='ppo_bitcoin_checkpoint'
)

class TradingMetricsCallback:
    def __init__(self, eval_env, log_freq=5000):
        self.eval_env = eval_env
        self.log_freq = log_freq
        self.episode_rewards = []
        self.portfolio_values = []

    def __call__(self, locals_dict, globals_dict):
        if locals_dict['self'].num_timesteps % self.log_freq == 0:
            obs = self.eval_env.reset()
            reward_total, steps = 0, 0
            for _ in range(100):
                action, _ = locals_dict['self'].predict(obs, deterministic=True)
                obs, reward, done, info = self.eval_env.step(action)
                reward_total += reward[0]
                steps += 1
                if done[0]: break

            if info and len(info) > 0:
                portfolio_value = info[0].get('portfolio_value', 10000)
                sharpe = info[0].get('sharpe', 0)
                self.portfolio_values.append(portfolio_value)
                print(f"📊 Step {locals_dict['self'].num_timesteps}: "
                      f"Reward={reward_total:.2f}, "
                      f"Portfolio=${portfolio_value:.2f}, "
                      f"Sharpe={sharpe:.3f}")
        return True

trading_callback = TradingMetricsCallback(eval_env)
callback_list = CallbackList([eval_callback, checkpoint_callback])

# ===========================
# 7. TRAINING CONFIGURATION (UNCHANGED)
# ===========================
print("\n🎯 Training Configuration:")
TOTAL_TIMESTEPS = 100000
SAVE_MODEL_PATH = 'models/ppo_bitcoin_final'

# ===========================
# 8. BASELINE PERFORMANCE (UNCHANGED)
# ===========================
print("\n📊 Testing random baseline...")

def test_random_agent(env, n_episodes=5):
    rewards, portfolios = [], []
    for _ in range(n_episodes):
        obs = env.reset()
        total_reward = 0
        while True:
            action = env.action_space.sample()
            obs, reward, done, info = env.step([action])
            total_reward += reward[0]
            if done[0]:
                portfolios.append(info[0].get('portfolio_value', 10000))
                rewards.append(total_reward)
                break
    return np.mean(rewards), np.mean(portfolios), portfolios

random_reward, random_portfolio, portfolios = test_random_agent(eval_env)
print(f"🎲 Random Agent - Avg Reward: {random_reward:.2f}, "
      f"Portfolio: ${random_portfolio:.2f}, "
      f"Return: {((random_portfolio - 10000) / 10000) * 100:.2f}%")

# ===========================
# 9. START TRAINING (UNCHANGED)
# ===========================
print(f"\n🚀 Starting PPO Training...")
print(f"🧠 LSTM Status: {'Enabled' if lstm_model else 'Disabled'}")
start_time = datetime.now()

try:
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=ActionLoggingCallback(),
        tb_log_name="ppo_bitcoin_training",
        reset_num_timesteps=False,
        progress_bar=True
    )
except KeyboardInterrupt:
    print("\n⚠️ Training interrupted. Saving model...")

# ===========================
# 10. SAVE FINAL MODEL (UPDATED)
# ===========================
print("\n💾 Saving model...")

try:
    model.save(SAVE_MODEL_PATH)
    print(f"✅ Saved to {SAVE_MODEL_PATH}")

    with open('models/training_config.pkl', 'wb') as f:
        pickle.dump({
            'ppo_config': ppo_config,
            'total_timesteps': TOTAL_TIMESTEPS,
            'lstm_enabled': lstm_model is not None,  # Added LSTM info
            'random_baseline': {
                'avg_reward': random_reward,
                'avg_portfolio': random_portfolio
            },
            'training_duration': str(datetime.now() - start_time)
        }, f)
    print("✅ Training config saved.")
except Exception as e:
    print(f"❌ Save failed: {e}")

print(f"\n🏁 Training Complete! LSTM Integration: {'Active' if lstm_model else 'Disabled'}")
print("Proceed to Cell 11 for evaluation.")