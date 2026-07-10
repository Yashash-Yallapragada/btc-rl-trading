# Cell 11: Agent Training with Monitoring
# Purpose: Train agent with performance tracking
# AI Coding Focus: Callbacks, logging, checkpointing
# Expected Output: Converged agent with positive rewards

import numpy as np
import matplotlib.pyplot as plt
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, CallbackList, BaseCallback
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.logger import configure
import os
import pandas as pd
from datetime import datetime
import pickle
import json
from collections import deque
import time

# Import environment
from updated_rl_env import BitcoinTradingEnv
from environment_setup import SEED

print("=" * 80)
print("CELL 11: AGENT TRAINING WITH MONITORING")
print("=" * 80)

# Set seed for reproducibility
set_random_seed(SEED)

# ===========================
# 1. ENHANCED CALLBACKS
# ===========================

class TradingPerformanceCallback(BaseCallback):
    """Enhanced callback for trading-specific metrics tracking"""
    
    def __init__(self, eval_env, log_freq=1000, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.log_freq = log_freq
        self.episode_rewards = []
        self.portfolio_values = []
        self.sharpe_ratios = []
        self.episode_lengths = []
        self.best_reward = -np.inf
        self.training_start_time = None
    
    def _on_training_start(self):
        self.training_start_time = time.time()
        print(f"Training started at {datetime.now().strftime('%H:%M:%S')}")
        
    def _on_step(self) -> bool:
        # CORE REQUIREMENT: TensorBoard logging for reward curves
        if self.n_calls % self.log_freq == 0:
            # Quick evaluation run
            obs = self.eval_env.reset()
            episode_reward = 0
            steps = 0
            
            for _ in range(200):  # Max episode length
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.eval_env.step(action)
                episode_reward += reward[0]
                steps += 1
                
                if done[0]:
                    break
            
            # Extract metrics from info
            if info and len(info) > 0:
                portfolio_value = info[0].get('portfolio_value', 100000)
                sharpe = info[0].get('sharpe', 0)
                max_drawdown = info[0].get('max_drawdown', 0)
                num_trades = info[0].get('num_trades', 0)
                
                # Store metrics
                self.episode_rewards.append(episode_reward)
                self.portfolio_values.append(portfolio_value)
                self.sharpe_ratios.append(sharpe)
                self.episode_lengths.append(steps)
                
                # Calculate portfolio return
                portfolio_return = ((portfolio_value - 100000) / 100000) * 100
                
                # CORE REQUIREMENT: TensorBoard logging
                self.logger.record('eval/episode_reward', episode_reward)
                self.logger.record('eval/portfolio_value', portfolio_value)
                self.logger.record('eval/portfolio_return_pct', portfolio_return)
                self.logger.record('eval/sharpe_ratio', sharpe)
                self.logger.record('eval/max_drawdown', max_drawdown)
                self.logger.record('eval/num_trades', num_trades)
                
                # Training speed tracking
                elapsed_time = time.time() - self.training_start_time
                steps_per_second = self.num_timesteps / elapsed_time if elapsed_time > 0 else 0
                self.logger.record('train/steps_per_second', steps_per_second)
                
                # Console logging
                print(f"[MONITOR] Step {self.num_timesteps:6d} | "
                      f"Reward: {episode_reward:7.2f} | "
                      f"Portfolio: ${portfolio_value:8.0f} ({portfolio_return:+.1f}%) | "
                      f"Sharpe: {sharpe:.3f} | "
                      f"Trades: {num_trades:3d}")
                
                # Track best performance
                if episode_reward > self.best_reward:
                    self.best_reward = episode_reward
                    print(f"New best reward: {episode_reward:.2f} at step {self.num_timesteps}")
        
        return True
    
    def get_training_metrics(self):
        """Return comprehensive training metrics"""
        return {
            'episode_rewards': self.episode_rewards,
            'portfolio_values': self.portfolio_values,
            'sharpe_ratios': self.sharpe_ratios,
            'episode_lengths': self.episode_lengths,
            'best_reward': self.best_reward
        }

# CORE REQUIREMENT: Early stopping if no improvement for 50k steps
class EarlyStoppingCallback(BaseCallback):
    """Stop training if no improvement for 50,000 steps"""
    
    def __init__(self, patience_steps=50000, min_improvement=0.1, verbose=0):
        super().__init__(verbose)
        self.patience_steps = patience_steps
        self.min_improvement = min_improvement
        self.best_mean_reward = -np.inf
        self.last_improvement_step = 0
        self.early_stop = False
        
    def _on_step(self) -> bool:
        # Check if we have evaluation results from EvalCallback
        if hasattr(self.parent, 'evaluations_results') and len(self.parent.evaluations_results) > 0:
            current_mean_reward = np.mean(self.parent.evaluations_results[-1])
            
            # Check for improvement
            if current_mean_reward > self.best_mean_reward + self.min_improvement:
                self.best_mean_reward = current_mean_reward
                self.last_improvement_step = self.num_timesteps
                if self.verbose > 0:
                    print(f"New best mean reward: {self.best_mean_reward:.2f} at step {self.num_timesteps}")
            
            # Check for early stopping
            steps_without_improvement = self.num_timesteps - self.last_improvement_step
            if steps_without_improvement >= self.patience_steps:
                if self.verbose > 0:
                    print(f"Early stopping triggered after {steps_without_improvement} steps without improvement")
                self.early_stop = True
                return False
                
        return True

# ===========================
# 2. DATA LOADING AND SETUP
# ===========================
print("Loading training data...")

X_train = np.load('data/processed/X_train.npy')
y_train = np.load('data/processed/y_train_clean.npy')  # Using cleaned data
X_test = np.load('data/processed/X_test.npy')
y_test = np.load('data/processed/y_test.npy')

# Clean data
if np.isnan(X_train).any():
    X_train = np.nan_to_num(X_train, nan=0.0)
if np.isnan(y_train).any():
    y_train = pd.Series(y_train).fillna(method='ffill').fillna(method='bfill').values

print(f"Training data: X_train {X_train.shape}, y_train {y_train.shape}")
print(f"Test data: X_test {X_test.shape}, y_test {y_test.shape}")

# Create directories
os.makedirs('models/monitored', exist_ok=True)
os.makedirs('logs/monitored', exist_ok=True)
os.makedirs('tensorboard_logs/monitored', exist_ok=True)

# ===========================
# 3. ENVIRONMENT CREATION
# ===========================
def make_env(X_data, y_data, rank=0, seed=SEED):
    """Factory to create environments"""
    def _init():
        env = BitcoinTradingEnv(
            X_data=X_data,
            y_data=y_data,
            window_size=60,
            initial_balance=100000,
            mode="train" if rank == 0 else "eval"
        )
        env.seed(seed + rank)
        return Monitor(env, filename=f'logs/monitored/monitor_{rank}.csv')
    return _init

train_env = DummyVecEnv([make_env(X_train, y_train, rank=0)])
eval_env = DummyVecEnv([make_env(X_test, y_test, rank=1)])

print("Environments created successfully")

# ===========================
# 4. PPO CONFIGURATION
# ===========================
print("Configuring PPO agent...")

ppo_config = {
    'policy': 'MlpPolicy',
    'env': train_env,
    'learning_rate': 5e-4,
    'n_steps': 1024,
    'batch_size': 128,
    'n_epochs': 15,
    'gamma': 0.995,
    'gae_lambda': 0.98,
    'clip_range': 0.15,
    'ent_coef': 0.02,
    'vf_coef': 0.5,
    'max_grad_norm': 0.5,
    'tensorboard_log': 'tensorboard_logs/monitored/',  # CORE REQUIREMENT
    'policy_kwargs': {
        'net_arch': [512, 256, 128],
        'activation_fn': torch.nn.ReLU
    },
    'verbose': 1,
    'seed': SEED
}

# Initialize PPO agent
model = PPO(**ppo_config)
print("PPO agent initialized with monitoring")

# ===========================
# 5. CALLBACK SETUP
# ===========================
print("Setting up callbacks...")

# Performance tracking callback
performance_callback = TradingPerformanceCallback(
    eval_env=eval_env,
    log_freq=1000,
    verbose=1
)

# CORE REQUIREMENT: Evaluation callback every 5k steps
# CORE REQUIREMENT: Save best model based on mean reward
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path='models/monitored/',
    log_path='logs/monitored/',
    eval_freq=5000,  # Every 5k steps as required
    deterministic=True,
    render=False,
    n_eval_episodes=5,
    verbose=1
)

# CORE REQUIREMENT: Early stopping callback
early_stopping = EarlyStoppingCallback(patience_steps=50000, verbose=1)

# Enhanced checkpoint callback with error handling
class SafeCheckpointCallback(CheckpointCallback):
    """Checkpoint callback with error handling"""
    
    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            try:
                path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps")
                self.model.save(path)
                if self.verbose > 0:
                    print(f"Model checkpoint saved at step {self.num_timesteps}")
            except Exception as e:
                print(f"Warning: Checkpoint save failed at step {self.num_timesteps}: {e}")
                # Try alternative save method
                try:
                    alt_path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_backup.pt")
                    torch.save(self.model.policy.state_dict(), alt_path)
                    print(f"Alternative checkpoint saved to: {alt_path}")
                except Exception as e2:
                    print(f"Alternative save also failed: {e2}")
        return True

checkpoint_callback = SafeCheckpointCallback(
    save_freq=10000,
    save_path='models/monitored/',
    name_prefix='ppo_monitored'
)

# Combine callbacks
callback_list = CallbackList([
    performance_callback,
    eval_callback,
    early_stopping,
    checkpoint_callback
])

print("Callbacks configured successfully")

# ===========================
# 6. BASELINE TESTING
# ===========================
print("Testing baseline strategies...")

def test_baseline_strategy(env, strategy='random', n_episodes=3):
    """Test baseline strategies"""
    results = []
    
    for episode in range(n_episodes):
        obs = env.reset()
        total_reward = 0
        step = 0
        
        while True:
            if strategy == 'random':
                action = np.array([env.action_space.sample()])
            elif strategy == 'buy_hold':
                action = np.array([1]) if step == 0 else np.array([0])
            else:  # always_hold
                action = np.array([0])
            
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
            step += 1
            
            if done[0]:
                final_info = info[0] if info and len(info) > 0 else {}
                results.append({
                    'total_reward': total_reward,
                    'portfolio_value': final_info.get('portfolio_value', 100000),
                    'return_pct': ((final_info.get('portfolio_value', 100000) - 100000) / 100000) * 100
                })
                break
    
    return results

# Test baselines
baseline_results = {}
for strategy in ['random', 'buy_hold', 'always_hold']:
    results = test_baseline_strategy(eval_env, strategy)
    avg_return = np.mean([r['return_pct'] for r in results])
    baseline_results[strategy] = {'avg_return_pct': avg_return, 'results': results}
    print(f"{strategy.replace('_', ' ').title()}: {avg_return:+.1f}% average return")

# ===========================
# 7. TRAINING EXECUTION
# ===========================
print("\n" + "=" * 80)
print("STARTING MONITORED TRAINING")
print("=" * 80)

TOTAL_TIMESTEPS = 100000  # Can be adjusted
TENSORBOARD_LOG_NAME = "monitored_training"

print(f"Training configuration:")
print(f"- Total timesteps: {TOTAL_TIMESTEPS:,}")
print(f"- TensorBoard logs: tensorboard_logs/monitored/{TENSORBOARD_LOG_NAME}")
print(f"- Evaluation frequency: 5,000 steps")
print(f"- Early stopping patience: 50,000 steps")
print(f"- Checkpoint frequency: 10,000 steps")

training_start = datetime.now()

try:
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callback_list,
        tb_log_name=TENSORBOARD_LOG_NAME,
        reset_num_timesteps=False,
        progress_bar=True
    )
    
    print("\nTraining completed successfully!")
    
except KeyboardInterrupt:
    print("\nTraining interrupted by user")
    
except Exception as e:
    print(f"\nTraining stopped due to error: {e}")

finally:
    training_duration = datetime.now() - training_start
    print(f"Training duration: {training_duration}")
    
    # ===========================
    # 8. SAVE RESULTS
    # ===========================
    print("\nSaving final model and results...")
    
    try:
        # Save final model with error handling
        final_model_path = 'models/monitored/ppo_final_monitored'
        model.save(final_model_path)
        print(f"Final model saved to: {final_model_path}")
    except Exception as e:
        print(f"Warning: Could not save final model: {e}")
        # Try alternative save
        try:
            torch.save(model.policy.state_dict(), 'models/monitored/policy_weights_final.pt')
            print("Policy weights saved as backup")
        except Exception as e2:
            print(f"Backup save also failed: {e2}")
    
    # Save training summary
    training_metrics = performance_callback.get_training_metrics()
    
    training_summary = {
        'training_duration': str(training_duration),
        'total_timesteps_completed': model.num_timesteps,
        'early_stopped': early_stopping.early_stop,
        'best_mean_reward': early_stopping.best_mean_reward,
        'best_performance_reward': performance_callback.best_reward,
        'baseline_results': baseline_results,
        'final_portfolio_values': training_metrics.get('portfolio_values', [])[-10:],
        'tensorboard_log_path': f'tensorboard_logs/monitored/{TENSORBOARD_LOG_NAME}',
        'config': {k: v for k, v in ppo_config.items() if k != 'env'}
    }
    
    with open('logs/monitored/training_summary.pkl', 'wb') as f:
        pickle.dump(training_summary, f)
    
    # ===========================
    # 9. TRAINING SUMMARY
    # ===========================
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"Duration: {training_duration}")
    print(f"Timesteps completed: {model.num_timesteps:,}")
    print(f"Early stopped: {early_stopping.early_stop}")
    print(f"Best mean reward: {early_stopping.best_mean_reward:.2f}")
    print(f"Best performance reward: {performance_callback.best_reward:.2f}")
    
    if training_metrics.get('portfolio_values'):
        final_portfolios = training_metrics['portfolio_values'][-5:]
        if final_portfolios:
            avg_final = np.mean(final_portfolios)
            final_return = ((avg_final - 100000) / 100000) * 100
            print(f"Average final portfolio: ${avg_final:.2f} ({final_return:+.1f}%)")
    
    print(f"\nFiles saved:")
    print(f"- models/monitored/ppo_final_monitored.zip")
    print(f"- logs/monitored/training_summary.pkl")
    print(f"- tensorboard_logs/monitored/{TENSORBOARD_LOG_NAME}")
    
    print(f"\nTo view TensorBoard logs:")
    print(f"tensorboard --logdir=tensorboard_logs/monitored/")
    
print("\n" + "=" * 80)
print("CELL 11 COMPLETE: Agent training with monitoring finished")
print("=" * 80)