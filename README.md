# Bitcoin RL Trading Agent

A reinforcement learning system for automated Bitcoin trading, built during a Summer 2025 internship at BeamX Techlabs. A custom Gymnasium trading environment simulates realistic portfolio management with transaction costs, and a Proximal Policy Optimization (PPO) agent is trained to make buy/sell/hold decisions based on technical indicators and market state.

The project combines feature engineering, deep reinforcement learning, rigorous walk-forward validation, and explainability analysis to evaluate whether RL can produce a statistically meaningful trading edge over Bitcoin's historical price data.

---

## Overview

- **Environment:** Custom Gymnasium-compatible `BitcoinTradingEnv`, simulating portfolio management with transaction costs, slippage, and position sizing constraints.
- **Features:** 10 engineered technical indicators — ALMA, Fisher Transform, Stochastic RSI, MACD, Bollinger Bands, ATR, EMA, OBV, CCI, Williams %R — alongside raw OHLCV data.
- **Agent:** PPO (Stable-Baselines3), with a dedicated hyperparameter search across 5 configurations and multiple training iterations (baseline → exploration-tuned → monitored training with early stopping and checkpointing).
- **Validation:** Anchored walk-forward validation across 5 independent chronological time windows and 3 random seeds (45 total trained agents), evaluated on 2 years of hourly BTC/USDT data (18,000 candles).
- **Reward design:** A 3-way ablation across reward function formulations, including a custom implementation of the differential Sharpe ratio (online risk-adjusted reward shaping, based on Moody & Saffell, 1998).
- **Explainability:** Feature importance via Random Forest surrogate modeling, permutation importance, and SHAP.
- **Side experiment:** An LSTM price-forecasting model, built and evaluated independently as a comparison point against the RL agent's decision-based approach.

---

## Results

Walk-forward validated performance across 5 independent test windows, 3 seeds, and 3 reward function variants (45 trained-and-evaluated agents total):

| Reward Function | Return (mean ± std) | Sharpe Ratio (mean ± std) | Max Drawdown (mean ± std) |
|---|---|---|---|
| Aggressive (2x-scaled, gain-weighted) | +3.4% ± 16.1% | 0.10 ± 0.44 | 7.1% ± 8.2% |
| Raw Return (symmetric) | +4.6% ± 16.1% | 0.15 ± 0.44 | 7.2% ± 7.9% |
| Differential Sharpe (online risk-adjusted) | +4.4% ± 13.8% | 0.13 ± 0.38 | 6.3% ± 5.6% |

**Key finding:** The agent's performance shows high variance across time periods (individual window returns ranged from -30% to +36%), and the differential Sharpe reward variant produced the most consistent (lowest-variance) risk-adjusted performance of the three formulations tested — evidence that explicit risk-adjusted reward shaping does meaningfully reduce outcome variance, even though a robust, generalizable trading edge was not established.

A dedicated diagnostic script (`src/evaluation/debug_environments.py`) was built specifically to sanity-check for common backtest red flags (unrealistically high Sharpe ratios, low volatility, unrealistic annualized returns) as part of the evaluation methodology.

Full breakdown and visualizations: `results/walk_forward_comparison.png`, `results/walk_forward_summary.csv`.

### LSTM price forecasting (side experiment)
- MSE: 0.0004, RMSE: 0.0194, MAPE: 1.73%, R²: 0.84
- Directional accuracy: 58%
- Not integrated into the RL pipeline — evaluated as an independent comparison of predictive vs. decision-based approaches to the same problem.

---

## Methodology

**1. Data pipeline** — Hourly BTC/USDT OHLCV data fetched via Binance's public market-data API, cleaned, validated, and enriched with technical indicators.

**2. Environment design** — State space combines a windowed sequence of market features with portfolio state (cash ratio, position, unrealized P&L). Discrete action space (hold/buy/sell). Reward reflects mark-to-market portfolio value change each step, with transaction costs and slippage applied on every trade.

**3. Agent training** — PPO with a tuned network architecture, multiple training iterations adding exploration scheduling and dynamic entropy adjustment to prevent premature convergence to a single action.

**4. Hyperparameter search** — 5 named configurations evaluated on Sharpe ratio, consistency, and convergence speed; best configuration retrained for the final model.

**5. Walk-forward validation** — Rather than a single train/test split, the model is retrained and re-evaluated across 5 chronologically-expanding windows, each tested on a genuinely unseen future period, with 3 random seeds per configuration for variance estimation.

**6. Feature importance** — Composite scoring across Random Forest feature importance, permutation importance, and SHAP values, identifying MACD and SMA(20) as the dominant decision-driving features.

---

## Repository structure

```
├── README.md
├── requirements.txt
├── docs/
│   └── project_report.pdf
├── src/
│   ├── data/
│   │   ├── fetcher.py                # Binance API client (retry/backoff/caching)
│   │   ├── fetch_extended_history.py # 2-year hourly data fetch
│   │   ├── indicators.py             # Technical indicator engineering
│   │   ├── preprocess.py             # Scaling, windowing, train/test splits
│   │   └── regenerate_processed_data.py  # Rebuilds X/y arrays from extended dataset
│   ├── env/
│   │   ├── updated_rl_env.py         # Gymnasium trading environment
│   │   ├── updated_rl_env_v2.py      # Extended env with configurable reward mode
│   │   ├── env_helper_portfolio_original.py
│   │   └── portfolio_manager_v2.py   # Portfolio simulation + 3-mode reward ablation
│   ├── training/
│   │   ├── train_rl_agent.py               # Baseline PPO training
│   │   ├── train_rl_agent_enhanced.py      # Exploration-scheduled variant
│   │   ├── train_monitored_agent.py        # Training with early stopping, checkpointing
│   │   ├── hyperparameter_search.py        # 5-configuration search
│   │   └── train_walk_forward.py           # Walk-forward validation orchestrator
│   ├── evaluation/
│   │   ├── evaluate_agent.py         # Sharpe/drawdown/win-rate evaluation vs benchmarks
│   │   ├── debug_environments.py     # Backtest sanity-check diagnostics
│   │   ├── diagnose_zero_trades.py   # Action-distribution diagnostic for training/eval behavior
│   │   ├── walk_forward_utils.py     # Walk-forward window generation
│   │   └── evaluate_walk_forward.py  # Results aggregation and comparison plots
│   ├── analysis/
│   │   └── feature_importance.py     # RF / permutation / SHAP composite scoring
│   └── lstm/
│       ├── lstm_model.py
│       ├── updated_rl_env_with_lstm.py
│       └── train_rl_agent_with_lstm.py
└── results/
    ├── walk_forward_summary.csv
    └── walk_forward_comparison.png
```

---

## How to run

```bash
pip install -r requirements.txt

# 1. Fetch data (2 years hourly BTCUSDT)
python src/data/fetch_extended_history.py

# 2. Compute technical indicators and preprocess
python src/data/indicators.py
python src/data/preprocess.py
python src/data/regenerate_processed_data.py

# 3. Run walk-forward validation (45 training runs)
python src/training/train_walk_forward.py

# 4. Aggregate and visualize results
python src/evaluation/evaluate_walk_forward.py
```

`train_walk_forward.py` supports automatic resume — if interrupted, it detects already-completed runs and continues from where it left off.

---

## Tech stack

Python · Stable-Baselines3 (PPO) · Gymnasium · PyTorch · TensorFlow/Keras · pandas · NumPy · scikit-learn · SHAP · Binance API

---

## Future work

- **DQN comparison** — training an off-policy DQN agent under identical conditions (same environment, data, and walk-forward windows) for a direct algorithm comparison against PPO.
- Reward function refinement to more strongly incentivize active position management.
- Multi-asset and multi-timeframe generalization testing.
- Real-time inference pipeline (data streaming and live model serving components included in `src/streaming/`, evaluated separately from the core backtesting study).

---

## Acknowledgements

Completed as part of an internship at BeamX Techlabs, under the mentorship of ShreeRam Dittakavi, CEO.
