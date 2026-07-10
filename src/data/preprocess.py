import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os

def preprocess_data(df, lookback=60, forecast_horizon=1):
    df = df.copy()

    # Drop unnecessary columns (like timestamp)
    df = df.drop(columns=['timestamp'], errors='ignore')

    # Fill missing values
    df = df.ffill().bfill()
    df = df.dropna()

    # Clip outliers using 1st and 99th percentiles
    df = df.clip(lower=df.quantile(0.01), upper=df.quantile(0.99), axis=1)

    # Normalize using MinMaxScaler
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df.values)

    # Create LSTM sequences
    X, y = [], []
    for i in range(lookback, len(scaled_data) - forecast_horizon):
        X.append(scaled_data[i - lookback:i])
        y.append(scaled_data[i + forecast_horizon - 1][0])  # Predicting 'close' price

    X = np.array(X)
    y = np.array(y)

    # Save full X and y arrays
    os.makedirs('data/processed', exist_ok=True)
    np.save('data/processed/X.npy', X)
    np.save('data/processed/y.npy', y)

    # Split into train, val, test (70/15/15)
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))

    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
    X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]

    # Save splits
    np.save('data/processed/X_train.npy', X_train)
    np.save('data/processed/y_train.npy', y_train)
    np.save('data/processed/X_val.npy', X_val)
    np.save('data/processed/y_val.npy', y_val)
    np.save('data/processed/X_test.npy', X_test)
    np.save('data/processed/y_test.npy', y_test)

    print(f"✅ Preprocessing done. Shapes - X: {X.shape}, y: {y.shape}")
    print("📁 Saved all arrays to 'data/processed/'")

if __name__ == "__main__":
    print("🚀 Starting preprocessing...")

    # Load cleaned dataset
    try:
        df = pd.read_csv('data/raw/binance_btcusdt.csv')
    except FileNotFoundError:
        print("❌ ERROR: 'data/raw/binance_btcusdt.csv' not found.")
        exit(1)

    # Run preprocessing
    preprocess_data(df)
    print("✅ All steps completed.")
