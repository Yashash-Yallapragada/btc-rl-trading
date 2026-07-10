# debug_environment.py
# Purpose: Debug the RL trading environment to identify unrealistic performance issues
# Run this to understand what's causing impossible Sharpe ratios

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from updated_rl_env import BitcoinTradingEnv
from environment_setup import SEED
import matplotlib.pyplot as plt

print("=" * 60)
print("ENVIRONMENT DEBUG ANALYSIS")
print("=" * 60)

# ===========================
# 1. LOAD MODEL AND DATA
# ===========================
print("Loading model and test data...")

# Load trained model
try:
    model = PPO.load('models/monitored/ppo_bitcoin_final_monitored')
    print("Model loaded successfully")
except:
    print("ERROR: Could not load model")
    exit()

# Load test data
X_test = np.load('data/processed/X_test.npy')
y_test = np.load('data/processed/y_test.npy')

# Convert 3D to 2D if needed
if X_test.ndim == 3:
    X_test = X_test[:, -1, :]
    print(f"Converted test data to: {X_test.shape}")

print(f"Test data: {X_test.shape}, Prices: {y_test.shape}")
print(f"Price range: ${y_test.min():.4f} - ${y_test.max():.4f}")

# ===========================
# 2. CREATE ENVIRONMENT
# ===========================
print("\nCreating environment...")

def make_debug_env():
    def _init():
        env = BitcoinTradingEnv(
            X_data=X_test,
            y_data=y_test,
            window_size=60,
            initial_balance=100000,
            mode="eval"
        )
        env.seed(SEED)
        return Monitor(env)
    return _init

env = DummyVecEnv([make_debug_env()])

# ===========================
# 3. RUN DEBUG EPISODE
# ===========================
print("\nRunning debug episode...")

obs = env.reset()
episode_data = {
    'portfolio_values': [100000],
    'rewards': [],
    'actions': [],
    'prices': [],
    'daily_returns': [],
    'step': 0
}

total_reward = 0
step = 0
max_steps = 50  # Limit for debugging

print(f"Starting debug run (max {max_steps} steps)...")

while step < max_steps:
    # Get model prediction
    action, _ = model.predict(obs, deterministic=True)
    episode_data['actions'].append(action[0])
    
    # Take step
    obs, reward, done, info = env.step(action)
    
    # Record data
    episode_data['rewards'].append(reward[0])
    total_reward += reward[0]
    
    if info and len(info) > 0:
        env_info = info[0]
        portfolio_val = env_info.get('portfolio_value', 100000)
        current_price = env_info.get('current_price', 0)
        
        episode_data['portfolio_values'].append(portfolio_val)
        episode_data['prices'].append(current_price)
        
        # Calculate daily return
        if len(episode_data['portfolio_values']) > 1:
            prev_val = episode_data['portfolio_values'][-2]
            daily_ret = (portfolio_val - prev_val) / prev_val
            episode_data['daily_returns'].append(daily_ret)
        else:
            episode_data['daily_returns'].append(0.0)
    
    step += 1
    
    if done[0]:
        break

print(f"Debug episode completed: {step} steps")

# ===========================
# 4. DETAILED ANALYSIS
# ===========================
print("\n" + "=" * 60)
print("DETAILED DIAGNOSTIC ANALYSIS")
print("=" * 60)

# Basic Statistics
portfolio_values = episode_data['portfolio_values']
rewards = episode_data['rewards']
daily_returns = episode_data['daily_returns']
actions = episode_data['actions']

print(f"\nEPISODE SUMMARY:")
print(f"- Steps taken: {step}")
print(f"- Total reward: {total_reward:.4f}")
print(f"- Initial portfolio: ${portfolio_values[0]:,.2f}")
print(f"- Final portfolio: ${portfolio_values[-1]:,.2f}")
print(f"- Total return: {((portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]):.4%}")

# Action Analysis
action_counts = {0: actions.count(0), 1: actions.count(1), 2: actions.count(2)}
print(f"\nACTION DISTRIBUTION:")
print(f"- Hold (0): {action_counts[0]} ({action_counts[0]/len(actions):.1%})")
print(f"- Buy (1): {action_counts[1]} ({action_counts[1]/len(actions):.1%})")
print(f"- Sell (2): {action_counts[2]} ({action_counts[2]/len(actions):.1%})")

# Reward Analysis
print(f"\nREWARD ANALYSIS:")
print(f"- Mean reward: {np.mean(rewards):.6f}")
print(f"- Std reward: {np.std(rewards):.6f}")
print(f"- Min reward: {np.min(rewards):.6f}")
print(f"- Max reward: {np.max(rewards):.6f}")
print(f"- Reward range: {np.max(rewards) - np.min(rewards):.6f}")

# Check for reward scaling issues
if np.mean(rewards) > 1.0:
    print("⚠️  WARNING: Average reward > 1.0 suggests reward scaling issues")
if np.std(rewards) < 0.001:
    print("⚠️  WARNING: Very low reward std suggests artificial consistency")

# Daily Return Analysis
if len(daily_returns) > 1:
    print(f"\nDAILY RETURN ANALYSIS:")
    print(f"- Mean daily return: {np.mean(daily_returns):.6f} ({np.mean(daily_returns)*100:.4f}%)")
    print(f"- Daily volatility: {np.std(daily_returns):.6f} ({np.std(daily_returns)*100:.4f}%)")
    print(f"- Min daily return: {np.min(daily_returns):.6f} ({np.min(daily_returns)*100:.4f}%)")
    print(f"- Max daily return: {np.max(daily_returns):.6f} ({np.max(daily_returns)*100:.4f}%)")
    
    # Annualized metrics
    annual_return = np.mean(daily_returns) * 252
    annual_vol = np.std(daily_returns) * np.sqrt(252)
    
    print(f"\nANNUALIZED METRICS:")
    print(f"- Annualized return: {annual_return:.4f} ({annual_return*100:.2f}%)")
    print(f"- Annualized volatility: {annual_vol:.4f} ({annual_vol*100:.2f}%)")
    
    # Calculate Sharpe ratio with proper risk-free rate
    risk_free_rate = 0.02  # 2% annual
    if annual_vol > 0:
        sharpe = (annual_return - risk_free_rate) / annual_vol
        print(f"- Sharpe ratio: {sharpe:.4f}")
        
        # Red flags
        if abs(sharpe) > 3.0:
            print("🚨 CRITICAL: Sharpe ratio > 3.0 is unrealistic for any trading strategy")
        if annual_return > 0.5:
            print("🚨 CRITICAL: >50% annual return is extremely suspicious")
        if annual_vol < 0.05:
            print("🚨 CRITICAL: <5% annual volatility is unrealistically low")
    else:
        print("- Sharpe ratio: undefined (zero volatility)")

# Portfolio Value Analysis
print(f"\nPORTFOLIO VALUE ANALYSIS:")
print(f"- Portfolio range: ${np.min(portfolio_values):,.2f} - ${np.max(portfolio_values):,.2f}")
print(f"- Max gain from start: {((np.max(portfolio_values) - portfolio_values[0]) / portfolio_values[0]):.4%}")
print(f"- Max loss from peak: {((np.min(portfolio_values) - np.max(portfolio_values)) / np.max(portfolio_values)):.4%}")

# Price Analysis
if episode_data['prices']:
    prices = episode_data['prices']
    print(f"\nPRICE MOVEMENT ANALYSIS:")
    print(f"- Price range: ${np.min(prices):.4f} - ${np.max(prices):.4f}")
    price_returns = [(prices[i] - prices[i-1])/prices[i-1] for i in range(1, len(prices))]
    if price_returns:
        print(f"- Mean price return: {np.mean(price_returns):.6f}")
        print(f"- Price volatility: {np.std(price_returns):.6f}")

# ===========================
# 5. VISUALIZATION
# ===========================
print(f"\nCreating diagnostic plots...")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Environment Debug Analysis', fontsize=16, fontweight='bold')

# Portfolio value over time
ax1 = axes[0, 0]
ax1.plot(portfolio_values, 'b-', linewidth=2)
ax1.axhline(y=100000, color='r', linestyle='--', alpha=0.7, label='Initial Value')
ax1.set_title('Portfolio Value Over Time')
ax1.set_xlabel('Steps')
ax1.set_ylabel('Portfolio Value ($)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Rewards over time
ax2 = axes[0, 1]
ax2.plot(rewards, 'g-', linewidth=1)
ax2.axhline(y=0, color='r', linestyle='--', alpha=0.7)
ax2.set_title('Rewards Over Time')
ax2.set_xlabel('Steps')
ax2.set_ylabel('Reward')
ax2.grid(True, alpha=0.3)

# Daily returns distribution
ax3 = axes[1, 0]
if len(daily_returns) > 1:
    ax3.hist(daily_returns, bins=20, alpha=0.7, edgecolor='black')
    ax3.axvline(x=np.mean(daily_returns), color='r', linestyle='--', label=f'Mean: {np.mean(daily_returns):.4f}')
    ax3.set_title('Daily Returns Distribution')
    ax3.set_xlabel('Daily Return')
    ax3.set_ylabel('Frequency')
    ax3.legend()
else:
    ax3.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax3.transAxes)
    ax3.set_title('Daily Returns Distribution')

# Actions over time
ax4 = axes[1, 1]
ax4.plot(actions, 'o-', markersize=4, linewidth=1)
ax4.set_title('Actions Over Time')
ax4.set_xlabel('Steps')
ax4.set_ylabel('Action (0=Hold, 1=Buy, 2=Sell)')
ax4.set_ylim(-0.5, 2.5)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('debug_environment_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# ===========================
# 6. FINAL DIAGNOSIS
# ===========================
print("\n" + "=" * 60)
print("FINAL DIAGNOSIS")
print("=" * 60)

# Check for common issues
issues_found = []

if len(daily_returns) > 1:
    if np.std(daily_returns) < 0.001:
        issues_found.append("Extremely low volatility - suggests artificial consistency")
    
    if np.mean(daily_returns) > 0.01:  # >1% daily return
        issues_found.append("Unrealistically high daily returns")
    
    annual_vol = np.std(daily_returns) * np.sqrt(252)
    if annual_vol > 0:
        sharpe = (np.mean(daily_returns) * 252 - 0.02) / annual_vol
        if abs(sharpe) > 3.0:
            issues_found.append(f"Impossible Sharpe ratio: {sharpe:.2f}")

if np.mean(rewards) > 5.0:
    issues_found.append("Reward values are too high - check reward scaling")

if action_counts[0] > 0.95 * len(actions):
    issues_found.append("Agent mostly holds - may not be learning to trade")

if issues_found:
    print("🚨 CRITICAL ISSUES IDENTIFIED:")
    for i, issue in enumerate(issues_found, 1):
        print(f"{i}. {issue}")
    
    print(f"\nRECOMMENDATIONS:")
    print("- Check your reward function in env_helper_portfolio.py")
    print("- Verify portfolio value calculations")
    print("- Ensure proper data preprocessing")
    print("- Review environment termination conditions")
    print("- Test on simpler scenarios with known expected outcomes")
else:
    print("✅ No obvious critical issues detected")
    print("Performance metrics appear within reasonable bounds")

print("\n" + "=" * 60)
print("DEBUG ANALYSIS COMPLETE")
print("=" * 60)