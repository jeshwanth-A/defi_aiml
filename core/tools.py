import json
import re
from datetime import datetime, timedelta

from langchain_core.tools import tool

from core.config import TOKENS
from core.data import (
    smart_fetch,
    preprocess_data,
    create_sequences,
    get_scaler_feature_columns,
)
from core.inference import run_inference, get_actual_price
from core.models import weights_exist, get_trained_models
from core.cache import save_prediction


def _normalize_token(token):
    if not token:
        return None
    t = str(token).strip().lower()
    aliases = {
        "btc": "bitcoin",
        "eth": "ethereum",
        "sol": "solana",
        "ada": "cardano",
        "uni": "uniswap",
    }
    t = aliases.get(t, t)
    return t if t in TOKENS else None


def _normalize_days(days, default=30):
    try:
        d = int(days)
    except Exception:
        d = default
    return max(1, d)


def _normalize_date(value):
    if not value:
        return datetime.now().date().isoformat()
    if isinstance(value, datetime):
        return value.date().isoformat()
    if hasattr(value, "strftime"):
        try:
            return value.strftime("%Y-%m-%d")
        except Exception:
            pass
    v = str(value).strip().lower()
    if v in ["today", "now"]:
        return datetime.now().date().isoformat()
    if v in ["tomorrow", "tommorrow", "tmr", "tmrw"]:
        return (datetime.now().date() + timedelta(days=1)).isoformat()
    v = re.sub(r"(?<=\d)(st|nd|rd|th)", "", v)
    v = re.sub(r"[,]+", " ", v)
    v = re.sub(r"\s+", " ", v).strip()

    for fmt in (
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%m/%d/%Y",
        "%d/%m/%Y",
        "%b %d %Y",
        "%B %d %Y",
        "%d %b %Y",
        "%d %B %Y",
    ):
        try:
            return datetime.strptime(v, fmt).date().isoformat()
        except Exception:
            continue
    return None


def _normalize_model(model):
    if not model:
        return "all"
    m = str(model).strip().lower()
    return m if m in ["pytorch", "tensorflow", "randomforest", "all"] else "all"


def _predict_with_models(token, days, target_date, model):
    if model != "all" and not weights_exist(model):
        return {
            "error": "Model weights not found",
            "model": model,
            "trained_models": get_trained_models(),
        }

    models = get_trained_models() if model == "all" else [model]
    if not models:
        return {
            "error": "No trained models. Use Train button first.",
            "trained_models": [],
        }

    df = smart_fetch(token, days + 10)
    if df is None or len(df) < 10:
        return {
            "token": token,
            "days": days,
            "target_date": target_date,
            "model": model,
            "results": [{"model": m, "error": "Not enough data"} for m in models],
        }

    try:
        df_proc, scaler = preprocess_data(df, fit_scaler=False)
    except FileNotFoundError:
        return {
            "token": token,
            "days": days,
            "target_date": target_date,
            "model": model,
            "results": [
                {"model": m, "error": "Scaler not found. Train first."} for m in models
            ],
        }
    except Exception as e:
        return {
            "token": token,
            "days": days,
            "target_date": target_date,
            "model": model,
            "results": [
                {"model": m, "error": f"Preprocessing failed: {str(e)[:120]}"}
                for m in models
            ],
        }

    feature_cols = get_scaler_feature_columns(scaler)
    missing = [c for c in feature_cols if c not in df_proc.columns]
    if missing:
        return {
            "token": token,
            "days": days,
            "target_date": target_date,
            "model": model,
            "results": [
                {
                    "model": m,
                    "error": f"Missing model features: {', '.join(missing)}",
                }
                for m in models
            ],
        }

    features = df_proc[feature_cols]
    X, _ = create_sequences(features, min(10, len(features) // 2))
    if len(X) == 0:
        return {
            "token": token,
            "days": days,
            "target_date": target_date,
            "model": model,
            "results": [{"model": m, "error": "No sequences"} for m in models],
        }

    actual = get_actual_price(token, target_date)

    results = []
    for m in models:
        try:
            pred = run_inference(m, X, scaler)
            if pred is None:
                results.append({"model": m, "error": "Inference failed"})
                continue

            mae = abs(pred - actual) if actual is not None else None
            save_prediction(token, m, days, target_date, float(pred), actual, mae)
            results.append(
                {
                    "model": m,
                    "predicted_usd": round(float(pred), 2),
                    "actual_usd": round(float(actual), 2)
                    if actual is not None
                    else None,
                    "mae": round(float(mae), 2) if mae is not None else None,
                }
            )
        except Exception as e:
            results.append({"model": m, "error": str(e)[:180]})

    return {
        "token": token,
        "days": days,
        "target_date": target_date,
        "model": model,
        "results": results,
    }


@tool
def list_supported_tokens() -> str:
    """Return the list of supported cryptocurrency token IDs."""
    return json.dumps({"tokens": TOKENS})


@tool
def check_model_status() -> str:
    """Check which ML models are trained and available for predictions."""
    return json.dumps(
        {
            "trained_models": get_trained_models(),
            "weights": {
                "pytorch": weights_exist("pytorch"),
                "tensorflow": weights_exist("tensorflow"),
                "randomforest": weights_exist("randomforest"),
            },
        }
    )


@tool
def get_current_price(token: str) -> str:
    """Get the current/latest USD price for a cryptocurrency token.

    Args:
        token: Token name like 'bitcoin', 'ethereum', 'solana', 'cardano', 'uniswap' (or btc, eth, sol, ada, uni)
    """
    t = _normalize_token(token)
    if not t:
        return json.dumps(
            {"error": f"Unsupported token '{token}'", "supported_tokens": TOKENS}
        )
    df = smart_fetch(t, 2)
    if df is None or len(df) == 0:
        return json.dumps({"error": "Could not fetch price data"})
    row = df.iloc[-1]
    return json.dumps(
        {
            "token": t,
            "price_usd": round(float(row["price"]), 2),
            "timestamp": str(row["timestamp"]),
        }
    )


@tool
def get_price_history(token: str, days: int = 30) -> str:
    """Get recent price history for a cryptocurrency token.

    Args:
        token: Token name like 'bitcoin', 'ethereum', 'solana', 'cardano', 'uniswap'
        days: Number of days of history (default 30)
    """
    t = _normalize_token(token)
    if not t:
        return json.dumps(
            {"error": f"Unsupported token '{token}'", "supported_tokens": TOKENS}
        )
    d = _normalize_days(days)
    df = smart_fetch(t, d)
    if df is None or len(df) == 0:
        return json.dumps({"error": "Could not fetch price data"})
    tail = df.tail(5)
    recent = [
        {
            "date": str(r["timestamp"].date()),
            "price_usd": round(float(r["price"]), 2),
        }
        for _, r in tail.iterrows()
    ]
    return json.dumps(
        {
            "token": t,
            "days_requested": d,
            "data_points": int(len(df)),
            "latest_price_usd": round(float(df.iloc[-1]["price"]), 2),
            "recent_prices": recent,
        }
    )


@tool
def predict_price(
    token: str, days: int = 30, target_date: str = "", model: str = "all"
) -> str:
    """Predict future price for a cryptocurrency token using trained ML models.

    Args:
        token: Token name like 'bitcoin', 'ethereum', 'solana', 'cardano', 'uniswap'
        days: Days of historical data to use (default 30)
        target_date: Date to predict for (YYYY-MM-DD format, or 'today', 'tomorrow')
        model: Which model to use - 'pytorch', 'tensorflow', 'randomforest', or 'all'
    """
    t = _normalize_token(token)
    if not t:
        return json.dumps(
            {"error": f"Unsupported token '{token}'", "supported_tokens": TOKENS}
        )
    d = _normalize_days(days)
    td = _normalize_date(target_date)
    if target_date and td is None:
        return json.dumps(
            {
                "error": f"Could not parse target_date '{target_date}'",
                "accepted_examples": [
                    "2026-03-23",
                    "2026/03/23",
                    "03/23/2026",
                    "Mar 23 2026",
                    "tomorrow",
                    "today",
                ],
            }
        )
    m = _normalize_model(model)
    result = _predict_with_models(t, d, td, m)
    return json.dumps(result)


ALL_TOOLS = [
    list_supported_tokens,
    check_model_status,
    get_current_price,
    get_price_history,
    predict_price,
]

TOOL_MAP = {t.name: t for t in ALL_TOOLS}
