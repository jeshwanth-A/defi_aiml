from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from .tools import get_price
from config import settings

async def lang_ai ( query , rag_context , file_context, memory, tool_use):

    llm = ChatGroq (
        api_key = settings.GROQ_API_KEY,
        model = settings.groq_model,
        temperature = 0.8
    )

    tools = [get_price]

    if tool_use:
        tool_section = """
        ## Tool: get_price
        You CAN fetch live cryptocurrency prices using the get_price tool. When someone asks about crypto prices, token prices, coin prices, or price history, use this tool.

        Parameters:
        - token_id: CoinGecko ID in lowercase (e.g., "bitcoin", "ethereum", "solana"). NOT the ticker symbol.
        - date: Number of days of history to fetch (e.g., 7 for last week, 30 for last month). NOT a calendar date.

        When NOT to use the tool: greetings, casual chat, general knowledge, non-crypto questions.
        If the user mentions a calendar date, convert it to number of days from today.
        Present results in a clean readable format. If the tool returns an error, tell the user and suggest checking the token name."""
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
        while True :
            response = await llm.ainvoke(messages)
            messages.append(response)
            if response.tool_calls:
                for tool_call in response.tool_calls:
                    if tool_call["name"] == "get_price":
                        result = await get_price.ainvoke(tool_call["args"])
                        messages.append(ToolMessage(content=result, tool_call_id=tool_call["id"]))
                continue
            else:
                return response.content
    else:
        return llm.astream(messages)






