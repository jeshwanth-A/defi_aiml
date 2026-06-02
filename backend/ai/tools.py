

from langchain_core.tools import tool
from ml import fetch_data

@tool
async def get_price(token_id: str, date: int) -> str :
    """Fetch live cryptocurrency price data for a given token over a specified number of days."""
    data = await fetch_data(token_id,date)
    return data

