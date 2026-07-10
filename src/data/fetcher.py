# fetcher.py

import os
import time
import requests
import pandas as pd
import json
from datetime import datetime

class BinanceDataFetcher:
    BASE_URL = "https://api.binance.com"

    def __init__(self, symbol="BTCUSDT", interval="1h", limit=1000, cache_dir="data/raw/"):
        self.symbol = symbol
        self.interval = interval
        self.limit = limit
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _get(self, endpoint, params, max_retries=5):
        """Robust GET request with retry on failure or rate limiting"""
        for attempt in range(max_retries):
            try:
                response = requests.get(f"{self.BASE_URL}{endpoint}", params=params, timeout=10)
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:
                    print("Rate limit hit, sleeping...")
                    time.sleep(2 ** attempt)
                else:
                    print(f"Error {response.status_code}: {response.text}")
                    break
            except Exception as e:
                print(f"Request error: {e}, retrying...")
                time.sleep(2 ** attempt)
        raise Exception("Failed to fetch data after retries.")

    def get_klines(self, start_time=None, end_time=None):
        """Fetch historical OHLCV data"""
        cache_file = os.path.join(self.cache_dir, f"{self.symbol}_{self.interval}_cached.csv")
        
        if os.path.exists(cache_file):
            print("📁 Using cached data...")
            df = pd.read_csv(cache_file, index_col="timestamp", parse_dates=True)
            return df

        params = {
            "symbol": self.symbol,
            "interval": self.interval,
            "limit": self.limit
        }
        if start_time:
            params["startTime"] = int(start_time)
        if end_time:
            params["endTime"] = int(end_time)

        data = self._get("/api/v3/klines", params)

        df = pd.DataFrame(data, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base", "taker_buy_quote", "ignore"
        ])

        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)

        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)

        df = df[["open", "high", "low", "close", "volume"]]

        # ✅ Validate
        if not self.validate_data(df):
            raise ValueError("❌ Data validation failed.")

        # ✅ Cache
        df.to_csv(cache_file)
        print(f"✅ Fetched and cached {len(df)} records.")
        return df

    def get_current_price(self):
        """Fetch current BTC price"""
        params = {"symbol": self.symbol}
        data = self._get("/api/v3/ticker/price", params)
        return float(data['price'])

    def validate_data(self, df):
        """Check for NaNs, zeros, or duplicate timestamps"""
        if df.isnull().sum().sum() > 0:
            print("❌ Missing values found.")
            return False
        if (df["volume"] == 0).any():
            print("⚠️ Zero volume rows found.")
        if df.index.duplicated().any():
            print("❌ Duplicate timestamps found.")
            return False
        return True

#Example use:
if __name__ == "__main__":
     fetcher = BinanceDataFetcher()
     df = fetcher.get_klines()
     print(df.head())
     print("Current price:", fetcher.get_current_price())
 # Save to CSV
df.to_csv("data/raw/binance_btcusdt.csv", index=False)
print("[INFO] Data saved to data/raw/binance_btcusdt.csv")