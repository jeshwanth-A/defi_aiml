import json
import re

from .tools import get_price, get_prices, predict_price

TOKEN_ALIASES = {
    "bitcoin": "bitcoin",
    "btc": "bitcoin",
    "ethereum": "ethereum",
    "ether": "ethereum",
    "eth": "ethereum",
    "solana": "solana",
    "sol": "solana",
    "dogecoin": "dogecoin",
    "doge": "dogecoin",
    "cardano": "cardano",
    "ada": "cardano",
    "ripple": "ripple",
    "xrp": "ripple",
}


def _detect_token(query: str) -> str | None:
    lowered = query.lower()
    for alias, token_id in sorted(TOKEN_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        if re.search(rf"\b{re.escape(alias)}\b", lowered):
            return token_id
    return None


def _money(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    return f"${value:,.2f}"


def _summarize_history(raw: str, token_id: str, days: int) -> str:
    data = json.loads(raw)
    if "error" in data:
        return f"I tried the live CoinGecko tool, but it returned: {data['error']}"

    prices = data.get("prices", [])
    if not prices:
        return f"I could not find price points for {token_id} over the last {days} days."

    first_price = prices[0][1]
    latest_price = prices[-1][1]
    high_price = max(point[1] for point in prices)
    low_price = min(point[1] for point in prices)

    return (
        f"Fetched live CoinGecko data for {token_id} over the last {days} days. "
        f"Latest price: {_money(latest_price)}. "
        f"Start price: {_money(first_price)}. "
        f"Range: {_money(low_price)} to {_money(high_price)}. "
        "The hosted AI provider is temporarily unavailable, so this response is coming from the backend tool fallback."
    )


def _summarize_exact_price(raw: str, token_id: str) -> str:
    data = json.loads(raw)
    if "error" in data:
        return f"I tried the live CoinGecko tool, but it returned: {data['error']}"

    return (
        f"Fetched live CoinGecko data for {token_id} on {data.get('requested_date') or data.get('date')}. "
        f"Price: {_money(data.get('price_usd'))}. "
        f"Market cap: {_money(data.get('market_cap_usd'))}. "
        f"Volume: {_money(data.get('total_volume_usd'))}. "
        "The hosted AI provider is temporarily unavailable, so this response is coming from the backend tool fallback."
    )


def _loads_json(raw: str) -> dict | None:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _extract_date(query: str) -> str | None:
    patterns = [
        r"\b(\d{1,2}[-/]\d{1,2}[-/]\d{4})\b",
        r"\b(\d{4}[-/]\d{1,2}[-/]\d{1,2})\b",
        r"\b(\d{1,2}(?:st|nd|rd|th)?\s+[a-zA-Z]+\s+\d{4})\b",
        r"\b([a-zA-Z]+\s+\d{1,2}(?:st|nd|rd|th)?\s+\d{4})\b",
        r"\b(\d{1,2}(?:st|nd|rd|th)?\s+[a-zA-Z]+)\b",
        r"\b([a-zA-Z]+\s+\d{1,2}(?:st|nd|rd|th)?)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, query)
        if match:
            return match.group(1)
    return None


async def fallback_response(query: str) -> str:
    token_id = _detect_token(query)
    lowered = query.lower()

    days_match = re.search(r"\b(?:last|past|previous)?\s*(\d{1,3})\s+days?\b", lowered)
    if token_id and days_match:
        days = max(1, min(int(days_match.group(1)), 365))
        raw = await get_prices.ainvoke({"token_id": token_id, "days": days})
        return _summarize_history(raw, token_id, days)

    exact_date = _extract_date(query)
    if token_id and exact_date and "price" in lowered:
        raw = await get_price.ainvoke({"token_id": token_id, "date": exact_date})
        return _summarize_exact_price(raw, token_id)

    if token_id == "ethereum" and any(word in lowered for word in ("predict", "forecast", "tomorrow", "next day", "next-day")):
        raw = await predict_price.ainvoke({})
        data = _loads_json(raw)
        if data is not None and "error" in data:
            return f"I tried the Ethereum prediction tool, but it returned: {data['error']}"

        return (
            "The hosted AI provider is temporarily unavailable, so this response is coming from "
            "the backend tool fallback. The experimental Ethereum LSTM model returned: "
            f"{raw.strip()} Treat this as a demo prediction, not financial advice."
        )

    if any(word in lowered for word in ("hello", "hi", "hey")):
        return (
            "Hi, I am the DeFi AI hosted demo. I can pull live crypto price data, "
            "for example: give last 30 days ethereum prices."
        )

    return (
        "The hosted AI provider is temporarily unavailable, but the backend is online. "
        "Try a tool-backed query like: give last 30 days ethereum prices."
    )
