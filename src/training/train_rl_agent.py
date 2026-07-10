# train_rl_agent_optimized.py

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

# Import optimized environment and seed
from updated_rl_env import BitcoinTradingEnv
from environment_setup import SEED

# Set seed
set_random_seed(SEED)

print("🚀 Starting OPTIMIZED RL Agent Training Setup")
print("=" * 60)

# ===========================
# 1. LOAD DATA
# ===========================
print("📊 Loading training data...")

X_train = np.load('data/processed/X_train.npy')
y_train = np.load('data/processed/y_train_clean.npy')
X_val = np.load('data/processed/X_val.npy')
y_val = np.load('data/processed/y_val.npy')
x_test = np.load('data/processed/X_test.npy')
y_test = np.load('data/processed/y_test.npy')

print("Checking for NaN values in training data...")
print(f"X_train NaN count: {np.isnan(X_train).sum()}")
print(f"y_train NaN count: {np.isnan(y_train).sum()}")

# Clean data
if np.isnan(X_train).any():
    print("Replacing NaN values in X_train with 0")
    X_train = np.nan_to_num(X_train, nan=0.0)
    
if np.isnan(y_train).any():
    print("Forward filling NaN values in y_train")
    y_train = pd.Series(y_train).fillna(method='ffill').fillna(method='bfill').values

if np.isnan(X_val).any():
    X_val = np.nan_to_num(X_val, nan=0.0)
    
if np.isnan(y_val).any():
    y_val = pd.Series(y_val).fillna(method='ffill').fillna(method='bfill').values

# Fix shapes if needed
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
# 2. CREATE ENVIRONMENTS
# ===========================
print("\n🏗️ Creating optimized training environments...")

os.makedirs('models', exist_ok=True)
os.makedirs('logs', exist_ok=True)
os.makedirs('tensorboard_logs', exist_ok=True)

def make_env(X_data, y_data, rank=0, seed=SEED):
    """Factory to create optimized environments"""
    def _init():
        env = BitcoinTradingEnv(
            X_data=X_data,
            y_data=y_data,
            window_size=60,
            initial_balance=100000,
            mode="train" if rank == 0 else "eval"
        )
        env.seed(seed + rank)
        env = Monitor(env)
        return env
    return _init

train_env = DummyVecEnv([make_env(X_train, y_train, rank=0)])
eval_env = DummyVecEnv([make_env(x_test, y_test, rank=1)])

print("✅ Optimized environments created successfully!")

# ===========================
# 3. OPTIMIZED PPO CONFIGURATION
# ===========================
print("\n⚙️ Configuring optimized PPO Agent...")

from torch import nn

# More aggressive PPO settings for better exploration and learning
ppo_config = {
    'policy': 'MlpPolicy',
    'env': train_env,
    'learning_rate': 5e-4,  # Slightly higher learning rate
    'n_steps': 1024,        # Reduced for faster updates
    'batch_size': 128,      # Larger batches
    'n_epochs': 15,         # More epochs per update
    'gamma': 0.995,         # Higher discount factor for longer-term rewards
    'gae_lambda': 0.98,     # Higher GAE lambda
    'clip_range': 0.15,     # Tighter clipping for stability
    'clip_range_vf': None,
    'ent_coef': 0.02,       # Higher entropy for more exploration
    'vf_coef': 0.5,
    'max_grad_norm': 0.5,
    'use_sde': False,
    'sde_sample_freq': -1,
    'target_kl': None,
    'tensorboard_log': 'tensorboard_logs/',
    'policy_kwargs': {
        'net_arch': [512, 256, 128],  # Larger network
        'activation_fn': nn.ReLU
    },
    'verbose': 1,
    'seed': SEED
}

print("📋 Optimized PPO Configuration:")
for key, value in ppo_config.items():
    if key != 'env':
        print(f"   {key}: {value}")

# ===========================
# 4. INITIALIZE PPO AGENT
# ===========================
print("\n🤖 Initializing optimized PPO Agent...")

model = PPO(**ppo_config)

from stable_baselines3.common.callbacks import BaseCallback

class EnhancedLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(EnhancedLoggingCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.portfolio_values = []

    def _on_step(self) -> bool:
        if self.n_calls % 1000 == 0:
            print(f"[TRAIN DEBUG] Step: {self.n_calls}, Num Timesteps: {self.num_timesteps}")
            
            # Log some training metrics
            if len(self.locals.get('infos', [])) > 0:
                info = self.locals['infos'][0]
                if 'portfolio_value' in info:
                    portfolio_val = info['portfolio_value']
                    portfolio_ratio = portfolio_val / 100000
                    print(f"[PORTFOLIO] Current value: ${portfolio_val:.2f} ({portfolio_ratio:.3f}x)")
                    
        return True

print("✅ Optimized PPO Agent initialized successfully!")

# ===========================
# 5. ENHANCED CALLBACKS
# ===========================
print("\n📊 Setting up enhanced training callbacks...")

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path='models/',
    log_path='logs/',
    eval_freq=3000,  # More frequent evaluation
    deterministic=True,
    render=False,
    n_eval_episodes=3,  # Faster evaluation
    verbose=1
)

checkpoint_callback = CheckpointCallback(
    save_freq=5000,  # More frequent checkpoints
    save_path='models/',
    name_prefix='ppo_bitcoin_optimized'
)

callback_list = CallbackList([eval_callback, checkpoint_callback])

# ===========================
# 6. TRAINING CONFIGURATION
# ===========================
print("\n🎯 Optimized Training Configuration:")
TOTAL_TIMESTEPS = 200000  # More training steps
SAVE_MODEL_PATH = 'models/ppo_bitcoin_optimized_final'

# ===========================
# 7. BASELINE PERFORMANCE
# ===========================
print("\n📊 Testing random baseline...")

def test_random_agent(env, n_episodes=3):
    rewards, portfolios = [], []
    for _ in range(n_episodes):
        obs = env.reset()
        total_reward = 0
        step_count = 0
        while step_count < 500:  # Limit episode length for testing
            action = env.action_space.sample()
            obs, reward, done, info = env.step([action])
            total_reward += reward[0]
            step_count += 1
            if done[0]:
                portfolios.append(info[0].get('portfolio_value', 100000))
                rewards.append(total_reward)
                break
        if step_count >= 500:  # Episode didn't terminate naturally
            portfolios.append(info[0].get('portfolio_value', 100000))
            rewards.append(total_reward)
    return np.mean(rewards), np.mean(portfolios), portfolios

random_reward, random_portfolio, portfolios = test_random_agent(eval_env)
print(f"🎲 Random Agent - Avg Reward: {random_reward:.2f}, "
      f"Portfolio: ${random_portfolio:.2f}, "
      f"Return: {((random_portfolio - 100000) / 100000) * 100:.2f}%")

# ===========================
# 8. START OPTIMIZED TRAINING
# ===========================
print("\n🚀 Starting optimized PPO Training...")
start_time = datetime.now()

try:
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=EnhancedLoggingCallback(),
        tb_log_name="ppo_bitcoin_optimized",
        reset_num_timesteps=False,
        progress_bar=True
    )
except KeyboardInterrupt:
    print("\n⚠️ Training interrupted. Saving model...")

# ===========================
# 9. SAVE FINAL MODEL
# ===========================
print("\n💾 Saving optimized model...")

try:
    model.save(SAVE_MODEL_PATH)
    print(f"✅ Saved to {SAVE_MODEL_PATH}")

    with open('models/optimized_training_config.pkl', 'wb') as f:
        pickle.dump({
            'ppo_config': ppo_config,
            'total_timesteps': TOTAL_TIMESTEPS,
            'random_baseline': {
                'avg_reward': random_reward,
                'avg_portfolio': random_portfolio
            },
            'training_duration': str(datetime.now() - start_time),
            'optimizations_applied': [
                'Increased position sizing (80% max)',
                'Reduced transaction costs (0.05%)',
                'Enhanced reward function',
                'Improved starting point selection',
                'Better observation space',
                'Larger neural network',
                'More aggressive PPO settings'
            ]
        }, f)
    print("✅ Optimized training config saved.")
except Exception as e:
    print(f"❌ Save failed: {e}")

print("\n🏁 Optimized Training Complete!")
print("\nKey optimizations applied:")
print("- Increased max position size from 20% to 80%")
print("- Reduced transaction costs from 0.1% to 0.05%") 
print("- Enhanced reward function with 2x multiplier")
print("- Better starting point to avoid flat price regions")
print("- Improved observation space with portfolio state")
print("- Larger neural network (512-256-128)")
print("- More aggressive PPO hyperparameters")