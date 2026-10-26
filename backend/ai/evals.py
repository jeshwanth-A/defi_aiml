
import asyncio
from .agent import lang_ai

cases = [
    # one past date -> get_price
    {"q": "What was the price of Ethereum on 12 June 2025?", "tools": ["get_price"]},
    {"q": "Bitcoin price on 01-01-2024", "tools": ["get_price"]},
    {"q": "how much was solana on 5 march 2025", "tools": ["get_price"]},
    {"q": "BTC price on 10th Jan", "tools": ["get_price"]},
    {"q": "What was dogecoin worth on 2025-12-25?", "tools": ["get_price"]},
    {"q": "price of xrp on 15 August 2024", "tools": ["get_price"]},
    # last N days -> get_prices
    {"q": "Show me the last 7 days of ETH", "tools": ["get_prices"]},
    {"q": "Bitcoin price history for 30 days", "tools": ["get_prices"]},
    {"q": "How has Solana moved over the past 90 days?", "tools": ["get_prices"]},
    {"q": "cardano prices for the last two weeks", "tools": ["get_prices"]},
    {"q": "recent ethereum market data", "tools": ["get_prices"]},
    # ethereum tomorrow -> predict_price
    {"q": "Predict Ethereum tomorrow", "tools": ["predict_price"]},
    {"q": "what might ETH be tomorrow?", "tools": ["predict_price"]},
    {"q": "forecast next-day ether price", "tools": ["predict_price"]},
    {"q": "give me your model's prediction for ethereum tomorrow", "tools": ["predict_price"]},
    # needs two tools
    {"q": "What was ETH on 1 Jan 2026 and what will it be tomorrow?", "tools": ["get_price", "predict_price"]},
    {"q": "Show bitcoin's last 7 days and ethereum's price on 1 March 2025", "tools": ["get_prices", "get_price"]},
    {"q": "Compare ethereum's last 30 days with your forecast for tomorrow", "tools": ["get_prices", "predict_price"]},
    # no tool: chat, unsupported prediction, missing date
    {"q": "hi", "tools": []},
    {"q": "what is a blockchain?", "tools": []},
    {"q": "predict bitcoin tomorrow", "tools": []},
    {"q": "what will solana be next week?", "tools": []},
    {"q": "what was ETH on the 5th?", "tools": []},
    {"q": "kjfdlsjf", "tools": []},
    {"q": "thanks, that's all", "tools": []},
]

async def doevals(cases):
    score = 0
    failed = []
    for case in cases :
        try:
            response, tool_output = await lang_ai(case["q"], "", "", [], True, True)
        except Exception as e:
            tool_output = [f"error: {e}"]
        ideal_outputs = case["tools"]
        if set(tool_output) == set(ideal_outputs) :
            score += 1
        else:
            failed.append((case["q"], ideal_outputs, tool_output))
        await asyncio.sleep(20)
    return score, failed

if __name__ == "__main__":
    score, failed = asyncio.run(doevals(cases))
    print(f"score is {score}/{len(cases)}")
    for q, ideal, got in failed:
        print(f"FAIL {q!r}: wanted {ideal}, got {got}")
