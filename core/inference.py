import sqlite3
from datetime import datetime

import torch

from core.config import CACHE_DB, admin_log
from core.data import fetch_and_parse, save_price_data
from core.models import load_model


def inverse_transform_price(scaled_price, scaler):
    feature_names = getattr(scaler, "feature_names_in_", None)
    if feature_names is not None:
        cols = [str(c) for c in feature_names]
        price_idx = cols.index("price") if "price" in cols else 0
    else:
        price_idx = 0

    scaled_price = float(scaled_price)
    if hasattr(scaler, "data_min_") and hasattr(scaler, "data_max_"):
        price_min = float(scaler.data_min_[price_idx])
        price_max = float(scaler.data_max_[price_idx])
        return scaled_price * (price_max - price_min) + price_min

    if hasattr(scaler, "scale_") and hasattr(scaler, "min_"):
        scale = float(scaler.scale_[price_idx])
        offset = float(scaler.min_[price_idx])
        if scale == 0:
            raise ValueError("Invalid scaler scale for price feature")
        return (scaled_price - offset) / scale

    raise ValueError("Unsupported scaler format")


def run_inference(name, X, scaler):
    m = load_model(name, input_size=X.shape[2] if len(X.shape) > 2 else X.shape[1])
    if m is None:
        admin_log("MODEL", f"Could not load model: {name}")
        return None
    try:
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
    except Exception as e:
        admin_log("MODEL", f"Inference failed for {name}: {str(e)[:140]}")
        return None

    try:
        pred_usd = inverse_transform_price(pred_scaled, scaler)
    except Exception as e:
        admin_log("MODEL", f"Inverse transform failed for {name}: {str(e)[:140]}")
        return None

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
