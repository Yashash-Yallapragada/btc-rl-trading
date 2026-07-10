# train_rl_agent_enhanced.py
# Enhanced training with better exploration and longer training

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
import os
from datetime import datetime
import json
import torch

# Import your environment
from updated_rl_env import BitcoinTradingEnv

# Set seed
SEED = 42
set_random_seed(SEED)

print("🚀 Starting ENHANCED RL Agent Training")
print("=" * 60)

# ===========================
# 1. LOAD DATA
# ===========================
print("📊 Loading data...")

X_train = np.load('data/processed/X_train.npy')
y_train = np.load('data/processed/y_train_clean.npy')
X_test = np.load('data/processed/X_test.npy')
y_test = np.load('data/processed/y_test.npy')

# Handle 3D to 2D conversion
if len(X_train.shape) == 3:
    print(f"Converting 3D to 2D by taking last timestep...")
    X_train = X_train[:, -1, :]
    X_test = X_test[:, -1, :] if len(X_test.shape) == 3 else X_test

# Clean data
X_train = np.nan_to_num(X_train, nan=0.0) if np.isnan(X_train).any() else X_train
y_train = pd.Series(y_train).fillna(method='ffill').fillna(method='bfill').values if np.isnan(y_train).any() else y_train
X_test = np.nan_to_num(X_test, nan=0.0) if np.isnan(X_test).any() else X_test
y_test = pd.Series(y_test).fillna(method='ffill').fillna(method='bfill').values if np.isnan(y_test).any() else y_test

print(f"✅ Data loaded: X_train {X_train.shape}, X_test {X_test.shape}")

# ===========================
# 2. CREATE ENVIRONMENTS
# ===========================
print("\n🏗️ Creating environments...")

os.makedirs('models/enhanced', exist_ok=True)
os.makedirs('logs/enhanced', exist_ok=True)
os.makedirs('tensorboard_logs/enhanced', exist_ok=True)

def make_env(X_data, y_data, rank=0, seed=SEED):
    def _init():
        env = BitcoinTradingEnv(
            X_data=X_data,
            y_data=y_data,
            window_size=60,
            initial_balance=100000,
            mode="train" if rank == 0 else "eval"
        )
        env.seed(seed + rank)
        return Monitor(env, filename=f'logs/enhanced/monitor_{rank}.csv')
    return _init

train_env = DummyVecEnv([make_env(X_train, y_train, rank=0)])
eval_env = DummyVecEnv([make_env(X_test, y_test, rank=1)])

print(f"✅ Environments created")

# ===========================
# 3. ENHANCED CALLBACK
# ===========================
class EnhancedTradingCallback(BaseCallback):
    """Enhanced callback with action distribution monitoring"""
    
    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.action_counts = {0: 0, 1: 0, 2: 0}  # hold, buy, sell
        self.episode_count = 0
        self.best_return = -np.inf
        
    def _on_step(self) -> bool:
        # Track action distribution
        if 'actions' in self.locals:
            action = self.locals['actions'][0]
            self.action_counts[action] = self.action_counts.get(action, 0) + 1
        
        # Log every 10k steps
        if self.n_calls % 10000 == 0 and self.n_calls > 0:
            total_actions = sum(self.action_counts.values())
            if total_actions > 0:
                hold_pct = (self.action_counts[0] / total_actions) * 100
                buy_pct = (self.action_counts[1] / total_actions) * 100
                sell_pct = (self.action_counts[2] / total_actions) * 100
                
                print(f"\n[Step {self.num_timesteps}] Action Distribution:")
                print(f"  Hold: {hold_pct:.1f}%, Buy: {buy_pct:.1f}%, Sell: {sell_pct:.1f}%")
                
                # Check if model is exploring
                if hold_pct > 95:
                    print("  ⚠️ Model mostly holding - increasing exploration...")
                    # Dynamically increase entropy coefficient to encourage exploration
                    if hasattr(self.model, 'ent_coef'):
                        self.model.ent_coef = min(0.1, self.model.ent_coef * 1.5)
                        print(f"  Entropy coefficient increased to: {self.model.ent_coef}")
            
            # Reset counts
            self.action_counts = {0: 0, 1: 0, 2: 0}
        
        # Check for episode end
        if 'dones' in self.locals and self.locals['dones'][0]:
            if 'infos' in self.locals:
                info = self.locals['infos'][0]
                portfolio_value = info.get('portfolio_value', 100000)
                portfolio_return = ((portfolio_value - 100000) / 100000) * 100
                num_trades = info.get('num_trades', 0)
                
                self.episode_count += 1
                
                if self.episode_count % 10 == 0:  # Log every 10 episodes
                    print(f"\n[Episode {self.episode_count}] "
                          f"Return: {portfolio_return:+6.1f}%, "
                          f"Trades: {num_trades:3d}")
                
                # Save best model based on return
                if portfolio_return > self.best_return:
                    self.best_return = portfolio_return
                    self.model.save('models/enhanced/best_by_return')
                    print(f"  🏆 New best return: {portfolio_return:+.1f}%!")
        
        return True

# ===========================
# 4. AGGRESSIVE PPO CONFIG
# ===========================
print("\n⚙️ Configuring PPO with enhanced exploration...")

ppo_config = {
    'policy': 'MlpPolicy',
    'env': train_env,
    'learning_rate': lambda f: 5e-4 * f,  # Learning rate schedule
    'n_steps': 1024,  # Smaller for more frequent updates
    'batch_size': 32,  # Smaller batch size
    'n_epochs': 20,  # More epochs per update
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_range': lambda f: 0.2 * f,  # Decaying clip range
    'clip_range_vf': None,
    'ent_coef': 0.05,  # Higher entropy for more exploration
    'vf_coef': 0.5,
    'max_grad_norm': 0.5,
    'use_sde': False,  # Use State Dependent Exploration
    'sde_sample_freq': 4,  # Sample new noise every 4 steps
    'target_kl': 0.02,
    'tensorboard_log': 'tensorboard_logs/enhanced/',
    'policy_kwargs': {
        'net_arch': [dict(pi=[128, 64], vf=[128, 64])],  # Separate networks
        'activation_fn': torch.nn.Tanh,  # Different activation
        'log_std_init': -0.5,  # Higher initial exploration
    },
    'verbose': 1,
    'seed': SEED
}

print("📋 Enhanced Configuration:")
print(f"   Learning rate: Decaying from 5e-4")
print(f"   Entropy coefficient: 0.05 (high exploration)")
print(f"   Using SDE: True (state-dependent exploration)")
print(f"   Separate actor-critic networks")

# ===========================
# 5. INITIALIZE AND TRAIN
# ===========================
print("\n🤖 Initializing enhanced PPO agent...")

model = PPO(**ppo_config)

# Setup callbacks
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path='models/enhanced/',
    log_path='logs/enhanced/',
    eval_freq=5000,
    deterministic=True,
    n_eval_episodes=2,
    verbose=1
)

checkpoint_callback = CheckpointCallback(
    save_freq=25000,
    save_path='models/enhanced/',
    name_prefix='ppo_enhanced'
)

enhanced_callback = EnhancedTradingCallback(verbose=1)

callback_list = CallbackList([eval_callback, checkpoint_callback, enhanced_callback])

# ===========================
# 6. EXTENDED TRAINING
# ===========================
print("\n🚀 Starting extended training with enhanced exploration...")
print(f"   Total timesteps: 300,000 (3x longer)")
print(f"   Using high entropy for exploration")
print(f"   Dynamic entropy adjustment")

TOTAL_TIMESTEPS = 300000  # 3x longer training
start_time = datetime.now()

try:
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callback_list,
        tb_log_name="ppo_enhanced",
        reset_num_timesteps=True,
        progress_bar=True
    )
    print("\n✅ Training completed!")
    
except KeyboardInterrupt:
    print("\n⚠️ Training interrupted")
    
finally:
    # Save final model
    model.save('models/enhanced/ppo_enhanced_final')
    
    # ===========================
    # 7. COMPREHENSIVE EVALUATION
    # ===========================
    print("\n📊 Comprehensive Final Evaluation...")
    
    def detailed_evaluation(model, env, n_episodes=10):
        """Detailed evaluation with action tracking"""
        results = {
            'rewards': [],
            'returns': [],
            'trades': [],
            'actions_taken': [],
            'portfolio_values': []
        }
        
        for episode in range(n_episodes):
            obs = env.reset()
            done = False
            episode_reward = 0
            actions_in_episode = []
            step = 0
            
            while not done and step < 500:
                action, _ = model.predict(obs, deterministic=True)
                actions_in_episode.append(int(action[0]))
                obs, reward, done, info = env.step(action)
                episode_reward += reward[0]
                step += 1
                
                if done[0]:
                    portfolio_value = info[0].get('portfolio_value', 100000)
                    portfolio_return = ((portfolio_value - 100000) / 100000) * 100
                    num_trades = info[0].get('num_trades', 0)
                    
                    results['rewards'].append(float(episode_reward))
                    results['returns'].append(float(portfolio_return))
                    results['trades'].append(int(num_trades))
                    results['portfolio_values'].append(float(portfolio_value))
                    
                    # Count actions
                    action_counts = {0: 0, 1: 0, 2: 0}
                    for a in actions_in_episode:
                        action_counts[a] = action_counts.get(a, 0) + 1
                    results['actions_taken'].append(action_counts)
                    
                    print(f"  Episode {episode+1}: "
                          f"Return: {portfolio_return:+6.1f}%, "
                          f"Trades: {num_trades:3d}, "
                          f"Actions: H={action_counts[0]}, B={action_counts[1]}, S={action_counts[2]}")
                    break
        
        return results
    
    print("\nTest Set Detailed Evaluation:")
    test_results = detailed_evaluation(model, eval_env, n_episodes=10)
    
    # Calculate statistics
    mean_return = np.mean(test_results['returns'])
    std_return = np.std(test_results['returns'])
    mean_trades = np.mean(test_results['trades'])
    
    # Calculate action distribution
    total_actions = {0: 0, 1: 0, 2: 0}
    for episode_actions in test_results['actions_taken']:
        for action, count in episode_actions.items():
            total_actions[action] += count
    
    total = sum(total_actions.values())
    action_percentages = {k: (v/total)*100 if total > 0 else 0 for k, v in total_actions.items()}
    
    print(f"\n📈 Final Results:")
    print(f"   Mean Return: {mean_return:+6.1f}% ± {std_return:.1f}%")
    print(f"   Mean Trades: {mean_trades:.1f}")
    print(f"   Action Distribution:")
    print(f"     Hold: {action_percentages[0]:.1f}%")
    print(f"     Buy:  {action_percentages[1]:.1f}%")
    print(f"     Sell: {action_percentages[2]:.1f}%")
    
    # Save results
    final_results = {
        'mean_return': float(mean_return),
        'std_return': float(std_return),
        'mean_trades': float(mean_trades),
        'action_percentages': {str(k): float(v) for k, v in action_percentages.items()},
        'all_returns': [float(r) for r in test_results['returns']],
        'all_trades': [int(t) for t in test_results['trades']],
        'training_duration': str(datetime.now() - start_time),
        'total_timesteps': TOTAL_TIMESTEPS
    }
    
    with open('models/enhanced/final_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n✅ Results saved to: models/enhanced/final_results.json")
    
    # Check if model learned to trade
    if mean_trades < 1:
        print("\n⚠️ Warning: Model still not trading actively!")
        print("Consider:")
        print("  1. Training for even longer (500k+ steps)")
        print("  2. Modifying reward function in portfolio manager")
        print("  3. Adding reward shaping for taking actions")
    elif mean_trades > 10:
        print("\n✅ Success: Model learned to trade actively!")
    
    print("\n" + "=" * 60)
    print("🏁 Enhanced Training Complete!")
    print(f"   Models saved in: models/enhanced/")
    print("\nTo view training progress:")
    print("   tensorboard --logdir tensorboard_logs/enhanced/")
    print("=" * 60)