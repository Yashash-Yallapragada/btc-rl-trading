# portfolio_manager_v2.py
# ============================================================================
# Extension of env_helper_portfolio_original.py
#
# WHY THIS EXISTS:
# The original PortfolioManager (env_helper_portfolio_original.py) used an
# aggressive reward function (return% * 200, with a 5x gain-bonus multiplier
# and asymmetric loss penalties). This produced strong-looking backtest
# numbers (Sharpe ~4.8) that are very likely inflated by:
#   1. A short, single evaluation window (~140 steps)
#   2. Reward shaping that directly rewards volatility-chasing behavior
#
# This module adds a `reward_mode` parameter so the SAME environment can be
# run with three different reward designs, enabling a controlled ablation
# study rather than a one-off fix:
#
#   'aggressive'        - the original reward (kept for direct comparison)
#   'raw_return'         - plain portfolio return, no bonus multipliers,
#                          symmetric penalties, tight reward clipping
#   'differential_sharpe' - online/incremental Sharpe ratio reward shaping,
#                          based on Moody & Saffell (1998), "Reinforcement
#                          Learning for Trading". Rewards the agent directly
#                          for improving risk-adjusted return at each step,
#                          rather than raw portfolio value change.
#
# All three modes share identical position sizing, transaction costs, and
# risk guardrails (emergency stop) so that the ONLY variable being tested
# is the reward function itself.
# ============================================================================

import numpy as np
import pandas as pd

# Bumped when reward logic changes mid-experiment, so results collected
# under different formulas can always be distinguished and not silently
# treated as directly comparable.
REWARD_FORMULA_VERSION = "v2_turnover_incentive"


class PortfolioManager:
    def __init__(
        self,
        transaction_cost=0.0005,
        slippage_pct=0.0002,          # NEW: simulates imperfect fill price
        initial_balance=100000,
        max_position_pct=0.3,          # NEW default: less aggressive than 0.8
        position_scaling=True,
        reward_mode="raw_return",      # 'aggressive' | 'raw_return' | 'differential_sharpe'
        sharpe_eta=0.01,               # learning rate for differential Sharpe
    ):
        self.transaction_cost = transaction_cost
        self.slippage_pct = slippage_pct
        self.initial_balance = initial_balance
        self.max_position_pct = max_position_pct
        self.position_scaling = position_scaling
        self.reward_mode = reward_mode
        self.sharpe_eta = sharpe_eta
        self.reset()

    def reset(self):
        self.cash = float(self.initial_balance)
        self.position = 0  # 0=no position, 1=long, -1=short
        self.position_size = 0.0
        self.entry_price = None
        self.portfolio_history = [float(self.initial_balance)]
        self.returns = []
        self.trades = []
        self.total_reward = 0.0

        # Differential Sharpe ratio running statistics (Moody & Saffell)
        self._A = 0.0  # running estimate of mean return
        self._B = 0.0  # running estimate of mean squared return

    # ------------------------------------------------------------------
    # Utility methods (identical to original, kept for safety/consistency)
    # ------------------------------------------------------------------
    def _safe_divide(self, numerator, denominator, default=0.0):
        if denominator == 0 or np.isnan(denominator) or np.isnan(numerator):
            return default
        result = numerator / denominator
        return default if np.isnan(result) or np.isinf(result) else result

    def _apply_slippage(self, price, direction):
        """Simulate imperfect fill: buys fill slightly higher, sells slightly lower."""
        if direction == "buy":
            return price * (1 + self.slippage_pct)
        elif direction == "sell":
            return price * (1 - self.slippage_pct)
        return price

    def _calculate_portfolio_value(self, current_price):
        if np.isnan(current_price) or current_price <= 0:
            return max(float(self.cash), 0.0)
        if np.isnan(self.cash):
            self.cash = 0.0
        self.cash = max(self.cash, 0.0)

        if self.position == 0 or self.entry_price is None or self.entry_price <= 0:
            return float(self.cash)

        price_ratio = self._safe_divide(current_price, self.entry_price, 1.0)

        if self.position == 1:
            current_position_value = self.position_size * price_ratio
        elif self.position == -1:
            current_position_value = self.position_size * (2 - price_ratio)
        else:
            current_position_value = 0.0

        current_position_value = max(current_position_value, 0.0)
        total_value = self.cash + current_position_value
        total_value = max(total_value, 0.01)

        if np.isnan(total_value) or np.isinf(total_value):
            return max(float(self.cash), 0.01)
        return float(total_value)

    def _can_open_position(self, price, investment_amount):
        if self.cash <= 0 or investment_amount <= 0:
            return False
        if investment_amount > self.cash:
            return False
        if np.isnan(price) or price <= 0:
            return False
        if investment_amount < 50:
            return False
        return True

    def _open_position(self, price, step, trade_type):
        current_portfolio_value = self._calculate_portfolio_value(price)
        max_investment_by_portfolio = current_portfolio_value * self.max_position_pct
        max_investment_by_cash = self.cash * 0.95
        investment_amount = min(max_investment_by_portfolio, max_investment_by_cash)

        min_cash_buffer = self.initial_balance * 0.02
        available_cash_for_investment = self.cash - min_cash_buffer
        investment_amount = min(investment_amount, available_cash_for_investment)

        if not self._can_open_position(price, investment_amount):
            return

        # Apply slippage on entry
        fill_price = self._apply_slippage(price, "buy" if trade_type == "buy" else "sell")

        transaction_cost = investment_amount * self.transaction_cost
        net_investment = investment_amount - transaction_cost

        if net_investment <= 0 or (self.cash - investment_amount) < 0:
            return

        self.position_size = float(net_investment)
        self.cash = float(self.cash - investment_amount)
        self.entry_price = float(fill_price)

        if trade_type == "buy":
            self.position = 1
        elif trade_type == "sell":
            self.position = -1

        self.trades.append({
            "type": trade_type, "price": float(fill_price),
            "size": float(self.position_size), "step": step,
        })

    def _close_position(self, price, step, trade_type):
        if self.position == 0 or self.entry_price is None or self.position_size == 0:
            return
        if np.isnan(price) or price <= 0:
            return

        # Apply slippage on exit (opposite direction of entry)
        fill_price = self._apply_slippage(price, "sell" if self.position == 1 else "buy")

        if self.position == 1:
            pnl_pct = self._safe_divide(fill_price - self.entry_price, self.entry_price, 0.0)
        else:
            pnl_pct = self._safe_divide(self.entry_price - fill_price, self.entry_price, 0.0)

        pnl_pct = max(pnl_pct, -0.95)
        final_position_value = self.position_size * (1 + pnl_pct)
        final_position_value = max(final_position_value, 0.0)

        transaction_cost = final_position_value * self.transaction_cost
        proceeds = max(final_position_value - transaction_cost, 0.0)

        self.cash += proceeds
        self.cash = max(self.cash, 0.0)
        profit = proceeds - self.position_size

        self.trades.append({
            "type": trade_type, "price": float(fill_price),
            "profit": float(profit), "profit_pct": float(pnl_pct), "step": step,
        })

        self.position = 0
        self.position_size = 0.0
        self.entry_price = None

    # ------------------------------------------------------------------
    # REWARD FUNCTIONS — this is the core of the ablation study
    # ------------------------------------------------------------------
    def _reward_aggressive(self, prev_val, new_val, return_pct):
        """Original reward, kept for direct before/after comparison."""
        if any(np.isnan([prev_val, new_val, return_pct])) or any(np.isinf([prev_val, new_val, return_pct])):
            return 0.0
        if prev_val <= 0 or new_val <= 0:
            return -2.0

        base_reward = return_pct * 200
        portfolio_ratio = self._safe_divide(new_val, self.initial_balance, 1.0)

        if portfolio_ratio > 1.0:
            base_reward += (portfolio_ratio - 1.0) * 5.0
        elif portfolio_ratio < 0.9:
            base_reward -= (0.9 - portfolio_ratio) * 2.0

        drawdown = self.max_drawdown()
        if not np.isnan(drawdown) and drawdown > 0.3:
            base_reward -= 0.5 * drawdown
        if new_val < self.initial_balance * 0.2:
            base_reward -= 3.0

        final_reward = np.clip(base_reward, -5.0, 10.0)
        return 0.0 if np.isnan(final_reward) or np.isinf(final_reward) else float(final_reward)

    def _reward_raw_return(self, prev_val, new_val, return_pct):
        """
        Simplified, symmetric reward: no gain multiplier, no asymmetric
        bonuses. Should produce much more conservative, realistic backtest
        statistics than the aggressive version.
        """
        if any(np.isnan([prev_val, new_val, return_pct])) or any(np.isinf([prev_val, new_val, return_pct])):
            return 0.0
        if prev_val <= 0 or new_val <= 0:
            return -1.0

        base_reward = return_pct * 100  # standard scaling, no multiplier

        # Symmetric drawdown penalty (applies equally regardless of direction)
        drawdown = self.max_drawdown()
        if not np.isnan(drawdown) and drawdown > 0.15:
            base_reward -= 0.3 * drawdown

        if new_val < self.initial_balance * 0.3:
            base_reward -= 2.0

        final_reward = np.clip(base_reward, -2.0, 2.0)  # tight, symmetric clip
        return 0.0 if np.isnan(final_reward) or np.isinf(final_reward) else float(final_reward)

    def _reward_differential_sharpe(self, return_pct):
        """
        Differential Sharpe Ratio (Moody & Saffell, 1998).
        Rewards the agent for the marginal improvement in risk-adjusted
        return at each step, using exponentially-weighted running estimates
        of mean (A) and second moment (B) of returns:

            delta_A = R_t - A_{t-1}
            delta_B = R_t^2 - B_{t-1}
            D_t = (B_{t-1} * delta_A - 0.5 * A_{t-1} * delta_B) / (B_{t-1} - A_{t-1}^2)^1.5

        This is a more principled reward than raw return: it directly
        optimizes for consistency (Sharpe), not just magnitude of return,
        so an agent trained with it should be structurally less prone to
        the "lucky window" inflation seen with the aggressive reward.
        """
        if np.isnan(return_pct) or np.isinf(return_pct):
            return 0.0

        R = return_pct
        A_prev, B_prev = self._A, self._B

        delta_A = R - A_prev
        delta_B = (R ** 2) - B_prev

        denom = (B_prev - A_prev ** 2)
        if denom <= 1e-8:
            D_t = 0.0  # not enough variance history yet
        else:
            denom = denom ** 1.5
            D_t = (B_prev * delta_A - 0.5 * A_prev * delta_B) / denom

        # Update running estimates
        self._A = A_prev + self.sharpe_eta * delta_A
        self._B = B_prev + self.sharpe_eta * delta_B

        reward = np.clip(D_t * 10.0, -2.0, 2.0)  # scale for learning signal
        return 0.0 if np.isnan(reward) or np.isinf(reward) else float(reward)

    def _calculate_reward(self, prev_val, new_val, return_pct):
        if self.reward_mode == "aggressive":
            return self._reward_aggressive(prev_val, new_val, return_pct)
        elif self.reward_mode == "differential_sharpe":
            return self._reward_differential_sharpe(return_pct)
        else:  # 'raw_return' default
            return self._reward_raw_return(prev_val, new_val, return_pct)

    # ------------------------------------------------------------------
    # Main update (identical control flow to original)
    # ------------------------------------------------------------------
    def update(self, action, price, step):
        if np.isnan(price) or price <= 0:
            return -1.0, max(float(self.cash), 0.01)

        # ------------------------------------------------------------------
        # BUG FIX (found via diagnose_zero_trades.py):
        # prev_value used to be recomputed as
        #   self._calculate_portfolio_value(price)
        # using THIS step's price — the same price used for new_value below.
        # When the action was a no-op (e.g. SELL while already short), this
        # made prev_value == new_value on every such step, forcing reward
        # to be mathematically exactly 0 regardless of how much the price
        # actually moved since the last step. The deterministic eval policy
        # exploited this: open one position, then spam the same action
        # forever for a guaranteed zero reward (safer than paying
        # transaction costs to trade normally).
        #
        # The correct prev_value is the portfolio value as of the END of
        # the previous step — i.e. computed with the PREVIOUS price, while
        # any open position was correctly marked-to-market at that time.
        # That value is exactly self.portfolio_history[-1].
        # ------------------------------------------------------------------
        prev_value = self.portfolio_history[-1]

        if prev_value < self.initial_balance * 0.05:
            if self.position != 0:
                self._close_position(price, step, f"emergency_close_{self.position}")
            return -5.0, self._calculate_portfolio_value(price)

        # ------------------------------------------------------------------
        # TURNOVER INCENTIVE (added mid-experiment — see reward_formula_version):
        # Walk-forward runs 1-17 showed the agent converging to "open one
        # position, never close it" in nearly every case. Root cause: since
        # mark-to-market accounting already credits reward for an open
        # position's paper gains every step, closing and re-entering only
        # adds transaction costs with zero reward benefit — so "never
        # trade again" was the mathematically reward-optimal policy.
        #
        # This adds a small explicit bonus specifically when a position is
        # CLOSED at a profit this step, giving the agent an actual reason
        # to realize gains and look for new opportunities, rather than
        # passively holding one early bet for the entire episode.
        # ------------------------------------------------------------------
        trades_before = len(self.trades)

        if action == 1:  # Buy
            if self.position == -1:
                self._close_position(price, step, "close_short")
            if self.position != 1 and self.cash > self.initial_balance * 0.05:
                self._open_position(price, step, "buy")
        elif action == 2:  # Sell
            if self.position == 1:
                self._close_position(price, step, "close_long")
            if self.position != -1 and self.cash > self.initial_balance * 0.05:
                self._open_position(price, step, "sell")

        turnover_bonus = 0.0
        if len(self.trades) > trades_before:
            newest_trade = self.trades[-1]
            if "profit" in newest_trade and newest_trade["profit"] > 0:
                # Small, capped bonus — enough to matter, not enough to
                # dominate the underlying reward signal
                turnover_bonus = min(0.3, newest_trade["profit_pct"] * 3.0)

        new_value = self._calculate_portfolio_value(price)
        new_value = max(new_value, 0.01)

        self.portfolio_history.append(float(new_value))
        return_pct = self._safe_divide(new_value - prev_value, prev_value, 0.0)
        self.returns.append(float(return_pct))

        reward = self._calculate_reward(prev_value, new_value, return_pct)
        reward = reward + turnover_bonus
        self.total_reward += reward

        return float(reward), float(new_value)

    # ------------------------------------------------------------------
    # Metrics (identical to original)
    # ------------------------------------------------------------------
    def sharpe_ratio(self):
        if len(self.returns) < 2:
            return 0.0
        r = np.array(self.returns)
        r = r[~np.isnan(r)]
        if len(r) < 2:
            return 0.0
        std_r = np.std(r)
        if std_r == 0 or np.isnan(std_r):
            return 0.0
        mean_r = np.mean(r)
        if np.isnan(mean_r):
            return 0.0
        sharpe = mean_r / std_r * np.sqrt(252)
        return 0.0 if np.isnan(sharpe) or np.isinf(sharpe) else float(sharpe)

    def sortino_ratio(self):
        if len(self.returns) < 2:
            return 0.0
        r = np.array(self.returns)
        r = r[~np.isnan(r)]
        downside = r[r < 0]
        if len(downside) < 2 or np.std(downside) == 0:
            return 0.0
        sortino = np.mean(r) / np.std(downside) * np.sqrt(252)
        return 0.0 if np.isnan(sortino) or np.isinf(sortino) else float(sortino)

    def max_drawdown(self):
        if len(self.portfolio_history) < 2:
            return 0.0
        v = np.array(self.portfolio_history)
        v = v[~np.isnan(v)]
        if len(v) < 2:
            return 0.0
        peak = np.maximum.accumulate(v)
        drawdown = (peak - v) / peak
        max_dd = np.max(drawdown)
        return 0.0 if np.isnan(max_dd) or np.isinf(max_dd) else float(max_dd)

    def get_portfolio_metrics(self, current_price):
        final_value = self._calculate_portfolio_value(current_price)
        final_value = max(final_value, 0.01)
        total_return = self._safe_divide(final_value - self.initial_balance, self.initial_balance, -1.0)

        profit_trades = [t for t in self.trades if "profit" in t]
        win_rate = np.mean([t["profit"] > 0 for t in profit_trades]) if profit_trades else 0.0

        return {
            "final_value": float(final_value),
            "total_return": float(total_return),
            "returns": [float(r) for r in self.returns],
            "sharpe": self.sharpe_ratio(),
            "sortino": self.sortino_ratio(),
            "max_drawdown": self.max_drawdown(),
            "num_trades": len(profit_trades),
            "win_rate": float(win_rate),
            "avg_trade_return": float(np.mean([t["profit"] for t in profit_trades])) if profit_trades else 0.0,
            "portfolio_history": [float(h) for h in self.portfolio_history],
            "all_trades": self.trades,
            "cash": float(max(self.cash, 0.0)),
            "position": self.position,
            "position_size": float(self.position_size),
            "reward_mode": self.reward_mode,
            "reward_formula_version": REWARD_FORMULA_VERSION,
        }
