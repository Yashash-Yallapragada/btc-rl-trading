import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, EMAIndicator
from ta.volatility import BollingerBands
from ta.volume import OnBalanceVolumeIndicator
from datetime import datetime, timedelta
import os

# === Load the raw data ===
df = pd.read_csv("data/raw/binance_btcusdt.csv")

# === Generate synthetic timestamps (1-min interval) ===
start_time = datetime(2022, 1, 1)
df['timestamp'] = [start_time + timedelta(minutes=i) for i in range(len(df))]
df.set_index('timestamp', inplace=True)

# === Ensure all required columns exist and are in correct dtype ===
df = df.astype({
    'open': float,
    'high': float,
    'low': float,
    'close': float,
    'volume': float
})

# === Technical Indicators ===

# 1. RSI (Relative Strength Index)
df['rsi'] = RSIIndicator(close=df['close'], window=14).rsi()

# 2. MACD (Moving Average Convergence Divergence)
macd = MACD(close=df['close'])
df['macd'] = macd.macd()
df['macd_signal'] = macd.macd_signal()

# 3. EMA (Exponential Moving Average)
df['ema_20'] = EMAIndicator(close=df['close'], window=20).ema_indicator()
df['ema_50'] = EMAIndicator(close=df['close'], window=50).ema_indicator()

# 4. Bollinger Bands
bb = BollingerBands(close=df['close'], window=20, window_dev=2)
df['bb_upper'] = bb.bollinger_hband()
df['bb_lower'] = bb.bollinger_lband()
df['bb_width'] = df['bb_upper'] - df['bb_lower']

# 5. Stochastic Oscillator
stoch = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'], window=14, smooth_window=3)
df['stoch_k'] = stoch.stoch()
df['stoch_d'] = stoch.stoch_signal()

# 6. On-Balance Volume (OBV)
df['obv'] = OnBalanceVolumeIndicator(close=df['close'], volume=df['volume']).on_balance_volume()

# 7. Price Rate of Change (ROC)
df['roc'] = df['close'].pct_change(periods=12) * 100

# 8. ALMA (Arnaud Legoux Moving Average) — custom
def alma(series, window=9, offset=0.85, sigma=6):
    m = offset * (window - 1)
    s = window / sigma
    weights = np.exp(-((np.arange(window) - m) ** 2) / (2 * s * s))
    weights /= weights.sum()
    return series.rolling(window).apply(lambda x: np.dot(x, weights), raw=True)

df['alma'] = alma(df['close'])

# 9. Fisher Transform — custom
def fisher_transform(series, period=10):
    min_val = series.rolling(period).min()
    max_val = series.rolling(period).max()
    val = 2 * ((series - min_val) / (max_val - min_val) - 0.5)
    val = val.clip(-0.999, 0.999)
    fisher = 0.5 * np.log((1 + val) / (1 - val))
    return fisher

df['fisher'] = fisher_transform(df['close'])

# 10. Stochastic RSI
def stoch_rsi(series, window=14):
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=window).mean()
    avg_loss = pd.Series(loss).rolling(window=window).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))

    min_rsi = rsi.rolling(window).min()
    max_rsi = rsi.rolling(window).max()
    return (rsi - min_rsi) / (max_rsi - min_rsi)

df['stoch_rsi'] = stoch_rsi(df['close'])

# === Save processed indicators ===
os.makedirs("data/processed", exist_ok=True)
df.to_csv("data/processed/btc_indicators.csv")

print("✅ Technical indicators computed and saved to data/processed/btc_indicators.csv")
