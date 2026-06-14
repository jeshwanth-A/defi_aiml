from langchain_core.tools import tool
from ml import fetch_exact_date_data, fetch_price_history, predict_data
from .cache import read_cache , write_cache

@tool
async def get_price(token_id: str, date: str) -> str :
    """Fetch exact cryptocurrency price data for a token on a specific date. Date can be dd-mm-yyyy or a phrase like 10th Jan."""
    check, data = read_cache("get_price",{"token_id":token_id,"days":date})
    if check:
        return data
    data = await fetch_exact_date_data(token_id,date)
    write_cache("get_price",{"token_id":token_id,"days":date},data,36000)
    return data

@tool
async def get_prices(token_id:str, days: int) ->  str :
    """Fetch cryptocurrency price data for a given token over a specified number of days."""
    check, data = read_cache("get_prices",{"token_id":token_id,"days":days})
    if check:
        return data
    data = await fetch_price_history(token_id, days)
    write_cache("get_prices",{"token_id":token_id,"days":days},data,36000)
    return data

@tool
async def predict_price() -> str:
    """Predict tomorrow's Ethereum price using the saved LSTM model."""
    check, data = read_cache("predict_price",{"token_id":"ethereum"})
    if check:
        return data
    data = await predict_data()
    write_cache("predict_price",{"token_id":"ethereum"},data,36000)
    return data
