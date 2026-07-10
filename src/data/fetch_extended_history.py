# fetch_extended_history.py
# ============================================================================
# WHY THIS EXISTS:
# The original dataset (data/raw/binance_btcusdt.csv) had only 1000 hourly
# candles (~41 days) — far too little for credible walk-forward validation.
# This script pulls ~2 years of hourly BTCUSDT data instead, using Binance's
# public data-api mirror (data-api.binance.vision), which is NOT subject to
# the same regional trading-eligibility restrictions as api.binance.com
# (that restriction applies to the live trading API; this is a public
# market-data-only endpoint).
#
# Binance's klines endpoint caps each request at 1000 candles, so this
# script paginates backwards in time until it reaches the target history
# length, then saves in the exact same CSV format your existing
# indicators.py / preprocess.py pipeline already expects — so nothing
# downstream needs to change.
# ============================================================================

import requests
import pandas as pd
import time
import os
from datetime import datetime, timedelta

BASE_URL = "https://data-api.binance.vision/api/v3/klines"
SYMBOL = "BTCUSDT"
INTERVAL = "1h"
TARGET_DAYS = 730          # ~2 years of hourly data

# Anchor the data to when the original project was actually done (July 2025),
# not today's date — keeps this consistent with the internship timeframe and
# avoids pulling in price action the original project never saw.
# Original data (BTCUSDT_1h_cached.csv) started 2025-06-17 15:00 and ran
# ~1000 hours forward (~41 days), so it ended around late July 2025.
END_DATE_STR = "2025-07-29"   # adjust if you know the exact original end date

MAX_CANDLES_PER_REQUEST = 1000
OUTPUT_PATH = "data/raw/binance_btcusdt_extended.csv"


def fetch_klines_page(end_time_ms, limit=1000):
    """Fetch one page of up to `limit` candles ending at end_time_ms."""
    params = {
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "endTime": end_time_ms,
        "limit": limit,
    }
    for attempt in range(5):
        try:
            resp = requests.get(BASE_URL, params=params, timeout=15)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                wait = 2 ** attempt
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  HTTP {resp.status_code}: {resp.text[:200]}")
                time.sleep(2 ** attempt)
        except requests.exceptions.RequestException as e:
            print(f"  Request failed (attempt {attempt+1}/5): {e}")
            time.sleep(2 ** attempt)
    raise RuntimeError("Failed to fetch klines after 5 retries")


def fetch_extended_history(target_days=TARGET_DAYS, end_date_str=END_DATE_STR):
    anchor_end = datetime.strptime(end_date_str, "%Y-%m-%d")

    print(f"Fetching ~{target_days} days of {INTERVAL} {SYMBOL} data...")
    print(f"Anchored to end date: {end_date_str} (matching original project timeframe)")
    print(f"Source: {BASE_URL}\n")

    all_candles = []
    end_time = int(anchor_end.timestamp() * 1000)
    target_start = int((anchor_end - timedelta(days=target_days)).timestamp() * 1000)

    page = 0
    while end_time > target_start:
        page += 1
        candles = fetch_klines_page(end_time, limit=MAX_CANDLES_PER_REQUEST)

        if not candles:
            print("No more data returned, stopping.")
            break

        all_candles = candles + all_candles  # prepend (going backwards in time)

        oldest_ts = candles[0][0]
        end_time = oldest_ts - 1  # next page ends just before this page's oldest candle

        oldest_date = datetime.fromtimestamp(oldest_ts / 1000).strftime("%Y-%m-%d %H:%M")
        print(f"  Page {page}: fetched {len(candles)} candles, "
              f"oldest so far = {oldest_date} | total so far = {len(all_candles)}")

        time.sleep(0.3)  # be polite to the API

        if page > 30:  # safety cap (30 pages * 1000 = 30k candles, plenty for 2 years hourly)
            print("Safety cap reached, stopping.")
            break

    print(f"\nTotal candles fetched: {len(all_candles)}")

    # Convert to DataFrame matching original format: timestamp,open,high,low,close,volume
    df = pd.DataFrame(all_candles, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "num_trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ])

    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    # Data quality checks (reusing the philosophy from your original fetcher.py)
    print("\nData quality checks:")
    print(f"  NaN values: {df.isna().sum().sum()}")
    print(f"  Duplicate timestamps: {df['timestamp'].duplicated().sum()}")
    print(f"  Zero-volume candles: {(df['volume'] == 0).sum()}")

    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)

    print(f"\nFinal dataset: {len(df)} candles")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    os.makedirs("data/raw", exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to: {OUTPUT_PATH}")

    return df


if __name__ == "__main__":
    fetch_extended_history()
