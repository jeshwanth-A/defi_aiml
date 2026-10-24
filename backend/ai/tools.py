from langchain_core.tools import tool
from ml import fetch_exact_date_data, fetch_price_history, predict_data
from .cache import read_cache , write_cache
import json

def is_error(data: str) -> bool:
    #dont cache failed calls like coingecko 429, or one bad call sticks for 10 hours
    try:
        return "error" in json.loads(data)
    except (json.JSONDecodeError, TypeError):
        return False

@tool
async def get_price(token_id: str, date: str) -> str :
    """Fetch exact cryptocurrency price data for a token on a specific date. Date can be dd-mm-yyyy or a phrase like 10th Jan."""
    check, data = read_cache("get_price",{"token_id":token_id,"days":date})
    if check:
        return data
    data = await fetch_exact_date_data(token_id,date)
    if not is_error(data):
        write_cache("get_price",{"token_id":token_id,"days":date},data,36000)
    return data

@tool
async def get_prices(token_id:str, days: int) ->  str :
    """Fetch cryptocurrency price data for a given token over a specified number of days."""
    check, data = read_cache("get_prices",{"token_id":token_id,"days":days})
    if check:
        return data
    data = await fetch_price_history(token_id, days)
    if not is_error(data):
        write_cache("get_prices",{"token_id":token_id,"days":days},data,36000)
    return data

@tool
async def predict_price() -> str:
    """Predict tomorrow's Ethereum price using the saved LSTM model."""
    cache_params = {"token_id": "ethereum", "model": "ethereum_lstm_v2"}
    check, data = read_cache("predict_price", cache_params)
    if check:
        return data
    data = await predict_data()
    if not is_error(data):
        write_cache("predict_price", cache_params, data, 36000)
    return data
