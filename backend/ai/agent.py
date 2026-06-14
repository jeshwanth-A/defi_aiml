from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from .tools import get_price, get_prices, predict_price
from config import settings

async def lang_ai ( query , rag_context , file_context, memory, tool_use):

    llm = ChatGroq (
        api_key = settings.GROQ_API_KEY,
        model = settings.groq_model,
        temperature = 0.8
    )

    tools = [get_price, get_prices, predict_price]

    if tool_use:
        tool_section = """
## Available Crypto Tools

You have three crypto tools. Choose the tool based on whether the user is asking for real historical data or the one supported future prediction.

### Tool: get_price
Use this tool when the user asks for actual cryptocurrency data on one specific past calendar date.

Parameters:
- token_id: CoinGecko coin ID in lowercase, e.g. "bitcoin", "ethereum", "solana". Do not use ticker symbols like BTC or ETH unless you convert them to CoinGecko IDs.
- date: Specific past date. Prefer dd-mm-yyyy when the user gives a year, e.g. "12-06-2025". If the user gives a day and month without a year, e.g. "10th Jan", pass that phrase as the date instead of inventing a year.

Use get_price for:
- "What was Ethereum price on 12 June 2025?"
- "Bitcoin price on 01-01-2024"
- "How much was Solana on this exact date?"

Do not use get_price for last N days or future prediction.

### Tool: get_prices
Use this tool when the user asks for actual recent/historical market data over the last N days.

Parameters:
- token_id: CoinGecko coin ID in lowercase.
- days: Number of past days to fetch, e.g. 7, 30, 90, 365.

Use get_prices for:
- "last 7 days of ETH"
- "Bitcoin price history for 30 days"
- "recent Solana market data"
- "show me past month crypto prices"

This returns actual CoinGecko historical data, not predictions.

### Tool: predict_price
Use this tool only when the user asks for Ethereum's tomorrow or next-day predicted price.

Parameters:
- This tool has no user-selectable coin parameter. It only predicts Ethereum.

Use predict_price for:
- "Predict Ethereum tomorrow"
- "Forecast next-day ETH price"
- "What might Ethereum be tomorrow?"

Important:
- predict_price is an experimental LSTM forecast for Ethereum only.
- Never present predicted price as guaranteed truth.
- Say clearly that it is a model prediction, not actual market data.
- This tool predicts only Ethereum's next daily price, not Bitcoin, Solana, arbitrary future dates, or multi-day forecasts.
- If the user asks to predict any coin other than Ethereum, explain that only Ethereum tomorrow prediction is available right now.

Tool choice rules:
- Specific past date -> get_price.
- Last N days / recent history -> get_prices.
- Ethereum tomorrow / Ethereum next day / ETH forecast -> predict_price.
- Non-Ethereum future prediction -> explain that only Ethereum tomorrow prediction is supported right now.
- Greetings, casual chat, or non-crypto questions -> do not use tools.
- If the user gives a ticker like ETH, convert it to CoinGecko ID "ethereum" when obvious.
- If the user gives a date with day and month but no year, pass the date phrase as written to get_price; the backend will normalize it.
- If the date is missing both month and year, ask for clarification instead of guessing.
"""
    else:
        tool_section = ""

    messages = [
        SystemMessage(content =f"""You are a helpful, friendly assistant. You can answer general questions, have casual conversations, and help with various topics.
{tool_section}
Rules:
- Be natural and conversational. Don't repeat the same response word for word.
- If someone asks what you can do, answer honestly based on your capabilities above.
- If input is truly random characters with no meaning (like "kjfdlsjf"), you can say "I didn't quite catch that — could you rephrase?" But for normal words, greetings, or partial sentences, respond naturally.
- Never say you can't do something you clearly can do."""),
        HumanMessage(content = f"here is query {query} here is context {rag_context} here is file context {file_context} here is the memory {memory}")
    ]


    if tool_use:
        llm = llm.bind_tools(tools)

        tool_map = {
            "get_price": get_price,
            "get_prices": get_prices,
            "predict_price": predict_price,
        }

        while True:
            response = await llm.ainvoke(messages)
            messages.append(response)

            if response.tool_calls:
                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]

                    if tool_name in tool_map:
                        result = await tool_map[tool_name].ainvoke(tool_call["args"])
                        messages.append(
                            ToolMessage(
                                content=result,
                                tool_call_id=tool_call["id"]
                            )
                        )
                continue
            else:
                return response.content
    else:
        return llm.astream(messages)






