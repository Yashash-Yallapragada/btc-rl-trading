# env_helper_portfolio_optimized.py - OPTIMIZED VERSION

import numpy as np
import pandas as pd

class PortfolioManager:
    def __init__(self, transaction_cost=0.0005, initial_balance=100000, max_position_pct=0.8, position_scaling=True):
        self.transaction_cost = transaction_cost  # Reduced from 0.001 to 0.0005
        self.initial_balance = initial_balance
        self.max_position_pct = max_position_pct  # Increased from 0.2 to 0.8
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
        # Validate inputs
        if np.isnan(current_price) or current_price <= 0:
            return max(float(self.cash), 0.0)
        
        if np.isnan(self.cash):
            self.cash = 0.0
        
        self.cash = max(self.cash, 0.0)
        
        if self.position == 0 or self.entry_price is None or self.entry_price <= 0:
            return float(self.cash)
        
        # Calculate position value
        price_ratio = self._safe_divide(current_price, self.entry_price, 1.0)
        
        if self.position == 1:  # Long position
            current_position_value = self.position_size * price_ratio
        elif self.position == -1:  # Short position  
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
        """Check if we can safely open a position"""
        if self.cash <= 0 or investment_amount <= 0:
            return False
        if investment_amount > self.cash:
            return False
        if np.isnan(price) or price <= 0:
            return False
        if investment_amount < 50:  # Reduced minimum from $100 to $50
            return False
        return True

    def _open_position(self, price, step, trade_type):
        """Open a new position with aggressive sizing for better returns"""
        current_portfolio_value = self._calculate_portfolio_value(price)
        
        # More aggressive position sizing
        max_investment_by_portfolio = current_portfolio_value * self.max_position_pct
        max_investment_by_cash = self.cash * 0.95  # Use up to 95% of cash
        
        investment_amount = min(max_investment_by_portfolio, max_investment_by_cash)
        
        # Minimum cash buffer reduced
        min_cash_buffer = self.initial_balance * 0.02  # Only 2% buffer
        available_cash_for_investment = self.cash - min_cash_buffer
        investment_amount = min(investment_amount, available_cash_for_investment)
        
        if not self._can_open_position(price, investment_amount):
            return
            
        transaction_cost = investment_amount * self.transaction_cost
        net_investment = investment_amount - transaction_cost
        
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
        """Close existing position"""
        if self.position == 0 or self.entry_price is None or self.position_size == 0:
            return
        
        if np.isnan(price) or price <= 0:
            return
            
        # Calculate P&L percentage
        if self.position == 1:  # Closing long
            pnl_pct = self._safe_divide(price - self.entry_price, self.entry_price, 0.0)
        else:  # Closing short
            pnl_pct = self._safe_divide(self.entry_price - price, self.entry_price, 0.0)
        
        # Allow larger losses but cap at -95%
        pnl_pct = max(pnl_pct, -0.95)
        
        final_position_value = self.position_size * (1 + pnl_pct)
        final_position_value = max(final_position_value, 0.0)
        
        transaction_cost = final_position_value * self.transaction_cost
        proceeds = final_position_value - transaction_cost
        proceeds = max(proceeds, 0.0)
        
        self.cash += proceeds
        self.cash = max(self.cash, 0.0)
        
        profit = proceeds - self.position_size
        
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
        """Optimized reward function focused on returns"""
        if any(np.isnan([prev_val, new_val, return_pct])) or any(np.isinf([prev_val, new_val, return_pct])):
            return 0.0
        
        if prev_val <= 0 or new_val <= 0:
            return -2.0
        
        # Primary reward: amplified returns
        base_reward = return_pct * 200  # Increased from 100 to 200
        
        # Portfolio ratio bonuses/penalties
        portfolio_ratio = self._safe_divide(new_val, self.initial_balance, 1.0)
        
        # Strong bonus for gains
        if portfolio_ratio > 1.0:
            gain_bonus = (portfolio_ratio - 1.0) * 5.0  # 5x multiplier for gains
            base_reward += gain_bonus
        
        # Moderate penalty for losses
        elif portfolio_ratio < 0.9:  # Only penalize after 10% loss
            loss_penalty = (0.9 - portfolio_ratio) * 2.0
            base_reward -= loss_penalty
        
        # Reduced drawdown penalty
        drawdown = self.max_drawdown()
        if not np.isnan(drawdown) and drawdown > 0.3:  # Only penalize extreme drawdown
            base_reward -= 0.5 * drawdown
            
        # Emergency penalty only for catastrophic losses
        if new_val < self.initial_balance * 0.2:  # 80% loss
            base_reward -= 3.0
        
        # Clip to wider range
        final_reward = np.clip(base_reward, -5.0, 10.0)  # Allow higher rewards
        
        if np.isnan(final_reward) or np.isinf(final_reward):
            return 0.0
            
        return float(final_reward)

    def update(self, action, price, step):
        """Optimized update function"""
        if np.isnan(price) or price <= 0:
            return -1.0, max(float(self.cash), 0.01)
    
        prev_value = self._calculate_portfolio_value(price)
        
        # Reduced emergency stop threshold
        if prev_value < self.initial_balance * 0.05:  # 95% loss
            if self.position != 0:
                self._close_position(price, step, f'emergency_close_{self.position}')
            return -5.0, self._calculate_portfolio_value(price)
    
        # Execute trading action
        if action == 0:  # Hold
            pass
        
        elif action == 1:  # Buy/Go Long
            if self.position == -1:
                self._close_position(price, step, 'close_short')
            if self.position != 1 and self.cash > self.initial_balance * 0.05:  # Reduced threshold
                self._open_position(price, step, 'buy')
            
        elif action == 2:  # Sell/Go Short  
            if self.position == 1:
                self._close_position(price, step, 'close_long')
            if self.position != -1 and self.cash > self.initial_balance * 0.05:  # Reduced threshold
                self._open_position(price, step, 'sell')
    
        new_value = self._calculate_portfolio_value(price)
        new_value = max(new_value, 0.01)
        
        self.portfolio_history.append(float(new_value))
        
        return_pct = self._safe_divide(new_value - prev_value, prev_value, 0.0)
        self.returns.append(float(return_pct))
        
        reward = self._calculate_reward(prev_value, new_value, return_pct)
        self.total_reward += reward
    
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
        final_value = max(final_value, 0.01)
        
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