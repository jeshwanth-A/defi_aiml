import json
import urllib.request
import pickle
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from core.config import admin_log, WEIGHTS_DIR, SCALER_PATH
from core.cache import get_cached_data, save_price_data


def fetch_and_parse(token_id="uniswap", days=30):
    url = f"https://api.coingecko.com/api/v3/coins/{token_id}/market_chart?vs_currency=usd&days={days}&interval=daily"
    admin_log("API", f"Fetching {token_id} ({days}d) from CoinGecko...")
    try:
        with urllib.request.urlopen(url) as r:
            data = json.loads(r.read().decode("utf-8"))
        prices, volumes = data.get("prices", []), data.get("total_volumes", [])
        if not prices:
            admin_log("API", f"Fetch {token_id}: NO DATA")
            return None
        df = pd.DataFrame(prices, columns=["timestamp", "price"])
        df["volume"] = [v[1] for v in volumes]
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        admin_log("API", f"Fetched {token_id}: {len(df)} rows")
        return df
    except Exception as e:
        admin_log("API", f"Fetch {token_id}: FAILED ({str(e)[:50]})")
        return None


def smart_fetch(token, days):
    cached = get_cached_data(token, days)
    if cached is not None and len(cached) >= days:
        return cached
    df = fetch_and_parse(token, days)
    if df is not None:
        save_price_data(token, df)
    return df


def preprocess_data(df, fit_scaler=True):
    df = df.copy()
    np.random.seed(42)
    df["sentiment"] = np.random.uniform(-1, 1, size=len(df))
    prices = df["price"].values
    ws = min(5, len(prices))
    if len(prices) >= ws:
        vol = np.std(np.lib.stride_tricks.sliding_window_view(prices, ws), axis=1)
        df["volatility"] = np.pad(vol, (ws - 1, 0), mode="edge")
    else:
        df["volatility"] = 0

    if fit_scaler:
        scaler = MinMaxScaler()
        df[["price", "volume", "sentiment", "volatility"]] = scaler.fit_transform(
            df[["price", "volume", "sentiment", "volatility"]]
        )
        os.makedirs(WEIGHTS_DIR, exist_ok=True)
        with open(SCALER_PATH, "wb") as f:
            pickle.dump(scaler, f)
    else:
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        df[["price", "volume", "sentiment", "volatility"]] = scaler.transform(
            df[["price", "volume", "sentiment", "volatility"]]
        )

    df["price_lag1"] = df["price"].shift(1)
    return df.dropna(), scaler


def create_sequences(data, seq_len=10):
    seq_len = min(seq_len, len(data) - 1)
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data.iloc[i : i + seq_len].values)
        y.append(data.iloc[i + seq_len]["price"])
    return np.array(X), np.array(y)
