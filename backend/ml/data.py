
import httpx
import json

async def fetch_data(token_id: str, date: int) -> str:
    url = f"https://api.coingecko.com/api/v3/coins/{token_id}/market_chart"
    params = {
        "vs_currency": "usd",
        "days": date,
        "interval": "daily"
    }
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params, timeout=10)
        data = response.json()
        if "prices" not in data:
            return json.dumps({"error": data.get("error", "check the token id can't fetch")})
        return json.dumps(data["prices"])
    except Exception as e:
        return json.dumps({"error": str(e)})
