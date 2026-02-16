import sqlite3
from datetime import datetime

import numpy as np
import torch

from core.config import CACHE_DB, admin_log
from core.data import fetch_and_parse, save_price_data
from core.models import load_model


def inverse_transform_price(scaled_price, scaler):
    dummy = np.array([[scaled_price, 0, 0, 0]])
    return scaler.inverse_transform(dummy)[0, 0]


def run_inference(name, X, scaler):
    m = load_model(name, input_size=X.shape[2] if len(X.shape) > 2 else X.shape[1])
    if m is None:
        return None
    if name == "pytorch":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        m = m.to(dev)
        with torch.no_grad():
            pred_scaled = (
                m(torch.tensor(X, dtype=torch.float32).to(dev))
                .cpu()
                .numpy()
                .flatten()[-1]
            )
    elif name == "tensorflow":
        pred_scaled = m.predict(X, verbose=0).flatten()[-1]
    elif name == "randomforest":
        pred_scaled = m.predict(X.reshape(X.shape[0], -1))[-1]
    else:
        return None

    pred_usd = inverse_transform_price(pred_scaled, scaler)
    admin_log(
        "MODEL",
        f"Inference {name}: {pred_scaled:.4f} (scaled) -> ${pred_usd:.2f} (USD)",
    )
    return pred_usd


def get_actual_price(token, target_date):
    target_dt = datetime.strptime(target_date, "%Y-%m-%d")
    if target_dt.date() > datetime.now().date():
        return None
    conn = sqlite3.connect(CACHE_DB)
    c = conn.cursor()
    c.execute(
        "SELECT price FROM price_data WHERE token = ? AND DATE(timestamp) = DATE(?) LIMIT 1",
        (token, target_date),
    )
    row = c.fetchone()
    conn.close()
    if row:
        return row[0]
    days_ago = (datetime.now() - target_dt).days + 5
    df = fetch_and_parse(token, min(days_ago, 365))
    if df is not None:
        save_price_data(token, df)
        df["date"] = df["timestamp"].dt.date
        match = df[df["date"] == target_dt.date()]
        if len(match) > 0:
            return match.iloc[0]["price"]
    return None
