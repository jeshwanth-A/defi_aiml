
import httpx
import json
import re
from datetime import date as date_type, datetime, timedelta, timezone

MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}

def normalize_history_date(value: str) -> str:
    raw = value.strip().lower()
    raw = re.sub(r"(\d+)(st|nd|rd|th)", r"\1", raw)
    raw = raw.replace(",", " ")
    raw = re.sub(r"\s+", " ", raw).strip()

    numeric = re.fullmatch(r"(\d{1,2})[-/](\d{1,2})[-/](\d{4})", raw)
    if numeric:
        day, month, year = numeric.groups()
        return f"{int(day):02d}-{int(month):02d}-{year}"

    iso = re.fullmatch(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", raw)
    if iso:
        year, month, day = iso.groups()
        return f"{int(day):02d}-{int(month):02d}-{year}"

    word = re.fullmatch(r"(\d{1,2})\s+([a-z]+)(?:\s+(\d{4}))?", raw)
    if not word:
        word = re.fullmatch(r"([a-z]+)\s+(\d{1,2})(?:\s+(\d{4}))?", raw)
        if word:
            month_name, day, year = word.groups()
        else:
            raise ValueError("Date must be dd-mm-yyyy, yyyy-mm-dd, or a format like 10 Jan 2026.")
    else:
        day, month_name, year = word.groups()

    month = MONTHS.get(month_name)
    if month is None:
        raise ValueError("Unknown month name. Use a date like 10 Jan 2026.")

    year = int(year) if year else date_type.today().year
    return f"{int(day):02d}-{month:02d}-{year}"

async def fetch_exact_date_range_data(token_id: str, normalized_date: str, requested_date: str) -> str:
    day = datetime.strptime(normalized_date, "%d-%m-%Y").replace(tzinfo=timezone.utc)
    next_day = day + timedelta(days=1)

    url = f"https://api.coingecko.com/api/v3/coins/{token_id}/market_chart/range"
    params = {
        "vs_currency": "usd",
        "from": int(day.timestamp()),
        "to": int(next_day.timestamp())
    }

    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params, timeout=10)

    response.raise_for_status()
    data = response.json()

    prices = data.get("prices", [])
    market_caps = data.get("market_caps", [])
    total_volumes = data.get("total_volumes", [])

    if not prices:
        return json.dumps({
            "error": "No market data found for this token/date. Check token_id or date."
        })

    return json.dumps({
        "token_id": token_id,
        "date": normalized_date,
        "requested_date": requested_date,
        "price_usd": prices[-1][1],
        "market_cap_usd": market_caps[-1][1] if market_caps else None,
        "total_volume_usd": total_volumes[-1][1] if total_volumes else None,
        "source": "market_chart_range"
    })

async def fetch_exact_date_data(token_id: str, date: str) -> str:
    #dd-mm-yyyy
    url = f"https://api.coingecko.com/api/v3/coins/{token_id}/history"
    try:
        normalized_date = normalize_history_date(date)
    except Exception as e:
        return json.dumps({"error": str(e)})

    params = {
        "date": normalized_date,
        "localization": "false"
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params, timeout=10)

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                return await fetch_exact_date_range_data(token_id, normalized_date, date)
            raise

        data = response.json()

        market_data = data.get("market_data")
        if not market_data:
            return json.dumps({
                "error": "No market data found for this token/date. Check token_id or date format dd-mm-yyyy."
            })

        current_price = market_data.get("current_price", {}).get("usd")
        market_cap = market_data.get("market_cap", {}).get("usd")
        total_volume = market_data.get("total_volume", {}).get("usd")

        return json.dumps({
            "token_id": token_id,
            "date": normalized_date,
            "requested_date": date,
            "price_usd": current_price,
            "market_cap_usd": market_cap,
            "total_volume_usd": total_volume
        })

    except Exception as e:
        return json.dumps({"error": str(e)})

async def fetch_price_history(token_id: str, days: int) -> str:
    """
    Fetch actual historical cryptocurrency price data for the last N days.
    Returns price, market cap, and volume points from CoinGecko.
    """
    url = f"https://api.coingecko.com/api/v3/coins/{token_id}/market_chart"
    params = {
        "vs_currency": "usd",
        "days": days,
        "interval": "daily"
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params, timeout=10)

        response.raise_for_status()
        data = response.json()

        if "prices" not in data:
            return json.dumps({
                "error": data.get("error", "Could not fetch price history. Check token_id.")
            })

        result = {
            "token_id": token_id,
            "days": days,
            "prices": data.get("prices", []),
            "market_caps": data.get("market_caps", []),
            "total_volumes": data.get("total_volumes", [])
        }

        return json.dumps(result)

    except Exception as e:
        return json.dumps({"error": str(e)})

async def fetch_model_window(token_id: str, days: int):
    """
    Fetch market data for model inference.
    Returns Python lists: prices, market_caps, volumes.
    """
    url = f"https://api.coingecko.com/api/v3/coins/{token_id}/market_chart"
    params = {
        "vs_currency": "usd",
        "days": days,
        "interval": "daily"
    }

    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params, timeout=10)

    response.raise_for_status()
    data = response.json()

    prices = [point[1] for point in data["prices"]]
    market_caps = [point[1] for point in data["market_caps"]]
    volumes = [point[1] for point in data["total_volumes"]]

    return prices, market_caps, volumes
