import sqlite3
from datetime import datetime

from core.config import CACHE_DB, admin_log


def init_cache():
    conn = sqlite3.connect(CACHE_DB)
    c = conn.cursor()
    c.execute(
        "CREATE TABLE IF NOT EXISTS price_data (id INTEGER PRIMARY KEY, token TEXT, timestamp TEXT, price REAL, volume REAL, fetched_at TEXT)"
    )
    c.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='predictions'"
    )
    if c.fetchone():
        c.execute("PRAGMA table_info(predictions)")
        cols = [row[1] for row in c.fetchall()]
        if "days" not in cols:
            c.execute("DROP TABLE predictions")
            admin_log(
                "CACHE",
                "Migrated: Dropped old predictions table (missing 'days' column)",
            )
    c.execute(
        "CREATE TABLE IF NOT EXISTS predictions (id INTEGER PRIMARY KEY, token TEXT, model TEXT, days INTEGER, target_date TEXT, predicted REAL, actual REAL, mae REAL, created_at TEXT)"
    )
    conn.commit()
    conn.close()


init_cache()


def save_price_data(token, df):
    conn = sqlite3.connect(CACHE_DB)
    c = conn.cursor()
    c.execute("DELETE FROM price_data WHERE token = ?", (token,))
    t = datetime.now().isoformat()
    for _, r in df.iterrows():
        c.execute(
            "INSERT INTO price_data (token, timestamp, price, volume, fetched_at) VALUES (?,?,?,?,?)",
            (token, str(r["timestamp"]), r["price"], r["volume"], t),
        )
    conn.commit()
    conn.close()
    admin_log("SAVE", f"Price data: {token} ({len(df)} rows)")


def get_cached_data(token, days):
    conn = sqlite3.connect(CACHE_DB)
    c = conn.cursor()
    c.execute(
        "SELECT timestamp, price, volume, fetched_at FROM price_data WHERE token = ? ORDER BY timestamp DESC LIMIT ?",
        (token, days),
    )
    rows = c.fetchall()
    conn.close()
    if not rows:
        admin_log("CACHE", f"Price data ({token}, {days}d): NOT FOUND")
        return None
    age = (datetime.now() - datetime.fromisoformat(rows[0][3])).total_seconds() / 3600
    if age > 24:
        admin_log("CACHE", f"Price data ({token}, {days}d): EXPIRED (age: {age:.1f}h)")
        return None
    admin_log(
        "CACHE",
        f"Price data ({token}, {days}d): FOUND (age: {age:.1f}h, {len(rows)} rows)",
    )
    import pandas as pd

    df = pd.DataFrame(rows, columns=["timestamp", "price", "volume", "fetched_at"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.drop("fetched_at", axis=1).sort_values("timestamp").reset_index(drop=True)


def get_cached_prediction(token, model, days, target_date):
    conn = sqlite3.connect(CACHE_DB)
    c = conn.cursor()
    c.execute(
        "SELECT predicted, actual, mae, created_at FROM predictions WHERE token=? AND model=? AND days=? AND target_date=? ORDER BY created_at DESC LIMIT 1",
        (token, model, days, target_date),
    )
    row = c.fetchone()
    conn.close()
    key = f"({token}, {model}, {days}d, {target_date})"
    if not row:
        admin_log("CACHE", f"Prediction {key}: NOT FOUND")
        return None
    age = (datetime.now() - datetime.fromisoformat(row[3])).total_seconds() / 3600
    if age > 24:
        admin_log("CACHE", f"Prediction {key}: EXPIRED (age: {age:.1f}h)")
        return None
    admin_log("CACHE", f"Prediction {key}: FOUND (age: {age:.1f}h)")
    return {"predicted": row[0], "actual": row[1], "mae": row[2]}


def save_prediction(token, model, days, target_date, predicted, actual, mae):
    conn = sqlite3.connect(CACHE_DB)
    c = conn.cursor()
    c.execute(
        "INSERT INTO predictions (token, model, days, target_date, predicted, actual, mae, created_at) VALUES (?,?,?,?,?,?,?,?)",
        (
            token,
            model,
            days,
            target_date,
            predicted,
            actual,
            mae,
            datetime.now().isoformat(),
        ),
    )
    conn.commit()
    conn.close()
    admin_log(
        "SAVE", f"Prediction: {token}/{model}/{days}d/{target_date} -> ${predicted:.2f}"
    )
