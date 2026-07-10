# env_helper_portfolio.py - CLEAN VERSION - No Debug Spam

import numpy as np
import pandas as pd

class PortfolioManager:
    def __init__(self, transaction_cost=0.001, initial_balance=100000, max_position_pct=0.2, position_scaling=True):
        self.transaction_cost = transaction_cost
        self.initial_balance = initial_balance
        self.max_position_pct = max_position_pct
        self.position_scaling = position_scaling
        self.reset()

    def reset(self):
        self.cash = float(self.initial_balance)
        self.position = 0  # 0=no position, 1=long, -1=short
        self.position_size = 0.0  # Dollar value of position
        self.entry_price = None
        self.portfolio_history = [float(self.initial_balance)]
        self.returns = []
        self.trades = []
        self.total_reward = 0.0

    def _safe_divide(self, numerator, denominator, default=0.0):
        """Safe division that handles NaN and zero cases"""
        if denominator == 0 or np.isnan(denominator) or np.isnan(numerator):
            return default
        result = numerator / denominator
        return default if np.isnan(result) or np.isinf(result) else result

    def _calculate_portfolio_value(self, current_price):
        """Calculate total portfolio value = cash + position value"""
        # Validate inputs silently
        if np.isnan(current_price) or current_price <= 0:
            return max(float(self.cash), 0.0)
        
        if np.isnan(self.cash):
            self.cash = 0.0
        
        # Ensure cash never goes negative
        self.cash = max(self.cash, 0.0)
        
        if self.position == 0 or self.entry_price is None or self.entry_price <= 0:
            return float(self.cash)
        
        # Calculate position value safely
        price_ratio = self._safe_divide(current_price, self.entry_price, 1.0)
        
        if self.position == 1:  # Long position
            current_position_value = self.position_size * price_ratio
        elif self.position == -1:  # Short position  
            current_position_value = self.position_size * (2 - price_ratio)
        else:
            current_position_value = 0.0
            
        # Ensure position value can't be negative (margin call protection)
        current_position_value = max(current_position_value, 0.0)
        
        total_value = self.cash + current_position_value
        
        # Final validation - portfolio value should never be negative
        total_value = max(total_value, 0.01)  # Minimum $0.01 to avoid division by zero
        
        if np.isnan(total_value) or np.isinf(total_value):
            return max(float(self.cash), 0.01)
            
        return float(total_value)

    def _can_open_position(self, price, investment_amount):
        """Check if we can safely open a position - silent validation"""
        if self.cash <= 0:
            return False
            
        if investment_amount <= 0:
            return False
            
        if investment_amount > self.cash:
            return False
            
        if np.isnan(price) or price <= 0:
            return False
            
        # Minimum trade size check
        if investment_amount < 100:  # Minimum $100 trade
            return False
            
        return True

    def _open_position(self, price, step, trade_type):
        """Open a new position - NO DEBUG PRINTS"""
        current_portfolio_value = self._calculate_portfolio_value(price)
        
        # Calculate investment amount - be more conservative
        max_investment_by_portfolio = current_portfolio_value * self.max_position_pct
        max_investment_by_cash = self.cash * 0.8  # Never use more than 80% of cash
        
        investment_amount = min(max_investment_by_portfolio, max_investment_by_cash)
        
        # Additional safety check - leave minimum cash buffer
        min_cash_buffer = self.initial_balance * 0.05  # 5% buffer
        available_cash_for_investment = self.cash - min_cash_buffer
        investment_amount = min(investment_amount, available_cash_for_investment)
        
        if not self._can_open_position(price, investment_amount):
            return
            
        transaction_cost = investment_amount * self.transaction_cost
        net_investment = investment_amount - transaction_cost
        
        # Final validation
        if net_investment <= 0 or (self.cash - investment_amount) < 0:
            return
        
        # Execute the trade
        self.position_size = float(net_investment)
        self.cash = float(self.cash - investment_amount)
        self.entry_price = float(price)
        
        if trade_type == 'buy':
            self.position = 1
        elif trade_type == 'sell':
            self.position = -1
            
        self.trades.append({
            'type': trade_type,
            'price': float(price),
            'size': float(self.position_size),
            'step': step
        })

    def _close_position(self, price, step, trade_type):
        """Close existing position - NO DEBUG PRINTS"""
        if self.position == 0 or self.entry_price is None or self.position_size == 0:
            return
        
        # Validate inputs silently
        if np.isnan(price) or price <= 0:
            return
            
        # Calculate P&L percentage
        if self.position == 1:  # Closing long
            pnl_pct = self._safe_divide(price - self.entry_price, self.entry_price, 0.0)
        else:  # Closing short
            pnl_pct = self._safe_divide(self.entry_price - price, self.entry_price, 0.0)
        
        # Cap maximum loss to prevent negative portfolio
        # For extreme cases, limit loss to 90% of position
        pnl_pct = max(pnl_pct, -0.9)
        
        # Calculate final position value
        final_position_value = self.position_size * (1 + pnl_pct)
        final_position_value = max(final_position_value, 0.0)  # Can't be negative
        
        transaction_cost = final_position_value * self.transaction_cost
        proceeds = final_position_value - transaction_cost
        proceeds = max(proceeds, 0.0)  # Proceeds can't be negative
        
        # Add proceeds back to cash
        self.cash += proceeds
        self.cash = max(self.cash, 0.0)  # Ensure cash stays non-negative
        
        # Calculate actual profit/loss
        profit = proceeds - self.position_size
        
        # Record trade
        self.trades.append({
            'type': trade_type,
            'price': float(price),
            'profit': float(profit),
            'profit_pct': float(pnl_pct),
            'step': step
        })
        
        # Reset position
        self.position = 0
        self.position_size = 0.0
        self.entry_price = None

    def _calculate_reward(self, prev_val, new_val, return_pct):
        """Calculate reward with focus on positive returns"""
        # Validate inputs
        if any(np.isnan([prev_val, new_val, return_pct])) or any(np.isinf([prev_val, new_val, return_pct])):
            return 0.0
        
        if prev_val <= 0 or new_val <= 0:
            return -1.0  # Heavy penalty for reaching zero/negative
        
        # Base reward from returns
        base_reward = return_pct * 100  # Scale up returns
        
        # Heavy penalty for portfolio decline
        portfolio_ratio = self._safe_divide(new_val, self.initial_balance, 1.0)
        if portfolio_ratio < 0.8:  # Lost more than 20%
            base_reward -= 2.0 * (0.8 - portfolio_ratio)
        
        # Bonus for maintaining portfolio value
        if portfolio_ratio >= 1.0:
            base_reward += 0.1
        
        # Penalty for excessive drawdown
        drawdown = self.max_drawdown()
        if not np.isnan(drawdown) and drawdown > 0.2:
            base_reward -= 1.0 * drawdown
            
        # Emergency penalty if portfolio is getting too low
        if new_val < self.initial_balance * 0.3:
            base_reward -= 5.0
        
        # Clip reward to reasonable range
        final_reward = np.clip(base_reward, -5.0, 2.0)
        
        if np.isnan(final_reward) or np.isinf(final_reward):
            return 0.0
            
        return float(final_reward)

    def update(self, action, price, step):
        """Main update function - ONLY logs every 500 steps"""
        # Validate inputs
        if np.isnan(price) or price <= 0:
            return -1.0, max(float(self.cash), 0.01)

        prev_value = self._calculate_portfolio_value(price)
        
        # Only log every 500 steps for faster training
        if step % 500 == 0:
            # Get LSTM prediction info if available
            lstm_info = ""
            if hasattr(self, 'current_env_ref') and hasattr(self.current_env_ref, 'lstm_model') and self.current_env_ref.lstm_model is not None:
                try:
                    lstm_preds = self.current_env_ref._get_lstm_predictions(self.current_env_ref.current_step)
                    predicted_price = lstm_preds[0]
                    price_change_pct = lstm_preds[1] * 100
                    lstm_info = f" | LSTM: ${predicted_price:.2f}({price_change_pct:+.1f}%)"
                except:
                    lstm_info = " | LSTM: Error"

            print(f"[UPDATE] Step {step}: Action={action}, Price=${price:.2f}{lstm_info}")
            print(f"[UPDATE] Before action - Cash: ${self.cash:.2f}, Position: {self.position}, Portfolio: ${prev_value:.2f}")
        
        # Emergency stop if portfolio is too low
        if prev_value < self.initial_balance * 0.1:
            if step % 500 == 0:
                print(f"[EMERGENCY STOP] Portfolio too low: ${prev_value:.2f}")
            if self.position != 0:
                self._close_position(price, step, f'emergency_close_{self.position}')
            return -5.0, self._calculate_portfolio_value(price)

        # Execute trading action - NO ACTION PRINTS (they slow down training)
        if action == 1:  # Buy/Go Long
            if self.position == -1:
                self._close_position(price, step, 'close_short')
            if self.position != 1 and self.cash > self.initial_balance * 0.1:
                self._open_position(price, step, 'buy')
        elif action == 2:  # Sell/Go Short
            if self.position == 1:
                self._close_position(price, step, 'close_long')
            if self.position != -1 and self.cash > self.initial_balance * 0.1:
                self._open_position(price, step, 'sell')

        # Calculate new portfolio value
        new_value = self._calculate_portfolio_value(price)
        new_value = max(new_value, 0.01)
        
        # Log results only every 500 steps
        if step % 500 == 0:
            action_names = ["HOLD", "BUY", "SELL"]
            print(f"[ACTION] {action_names[action]}")
            print(f"[UPDATE] After action - Cash: ${self.cash:.2f}, Position: {self.position}, Portfolio: ${new_value:.2f}")

        self.portfolio_history.append(float(new_value))

        # Calculate return
        return_pct = self._safe_divide(new_value - prev_value, prev_value, 0.0)
        self.returns.append(float(return_pct))

        # Calculate reward
        reward = self._calculate_reward(prev_value, new_value, return_pct)
        self.total_reward += reward

        if step % 500 == 0:
            print(f"[UPDATE] Return: {return_pct:.4f}, Reward: {reward:.4f}")
            print("-" * 60)

        return float(reward), float(new_value)

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
        final_value = max(final_value, 0.01)  # Ensure positive
        
        total_return = self._safe_divide(final_value - self.initial_balance, self.initial_balance, -1.0)
        
        profit_trades = [t for t in self.trades if 'profit' in t]
        win_rate = np.mean([t['profit'] > 0 for t in profit_trades]) if profit_trades else 0.0
        
        return {
            "final_value": float(final_value),
            "total_return": float(total_return),
            "returns": [float(r) for r in self.returns],
            "sharpe": self.sharpe_ratio(),
            "max_drawdown": self.max_drawdown(),
            "num_trades": len(profit_trades),
            "win_rate": float(win_rate),
            "avg_trade_return": float(np.mean([t['profit'] for t in profit_trades])) if profit_trades else 0.0,
            "portfolio_history": [float(h) for h in self.portfolio_history],
            "all_trades": self.trades,
            "cash": float(max(self.cash, 0.0)),
            "position": self.position,
            "position_size": float(self.position_size)
        }