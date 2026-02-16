import json
import os

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from core.config import GROQ_API_KEY, GROQ_MODEL, GROQ_TEMPERATURE
from core.tools import ALL_TOOLS, TOOL_MAP


SYSTEM_PROMPT = """You are DefiDoza, a helpful crypto forecasting assistant.

You have access to these tools:
- list_supported_tokens: Get the list of tokens you can work with
- check_model_status: See which ML models are trained
- get_current_price: Get current USD price for a token
- get_price_history: Get recent price history
- predict_price: Run ML prediction for a token's future price

Guidelines:
- When users ask about prices, use get_current_price
- When users ask for predictions/forecasts, use predict_price
- When users ask about trends or history, use get_price_history
- When users ask what tokens are supported, use list_supported_tokens
- When users ask about trained models, use check_model_status
- If prediction fails because no models are trained, tell the user to use the Train button first
- Be concise and helpful in your responses
- Format prices nicely (e.g., "$42,123.45")"""


class DefiDozaAgent:
    """Simple tool-calling agent using Groq directly (no AgentExecutor needed)."""

    def __init__(self, api_key: str):
        self.llm = ChatGroq(
            model=GROQ_MODEL,
            api_key=api_key,
            temperature=GROQ_TEMPERATURE,
        )
        self.llm_with_tools = self.llm.bind_tools(ALL_TOOLS)
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    def _execute_tool(self, tool_call) -> str:
        """Execute a tool call and return the result."""
        tool_name = tool_call["name"]
        tool_args = tool_call.get("args", {})

        if tool_name not in TOOL_MAP:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})

        try:
            result = TOOL_MAP[tool_name].invoke(tool_args)
            return result
        except Exception as e:
            return json.dumps({"error": str(e)})

    def ask(self, question: str) -> str:
        """Process a user question, potentially using tools."""
        self.messages.append({"role": "user", "content": question})

        response = self.llm_with_tools.invoke(self.messages)

        if hasattr(response, "tool_calls") and response.tool_calls:
            self.messages.append(response)

            for tool_call in response.tool_calls:
                tool_result = self._execute_tool(tool_call)
                self.messages.append(
                    ToolMessage(content=tool_result, tool_call_id=tool_call["id"])
                )

            final_response = self.llm_with_tools.invoke(self.messages)
            self.messages.append(final_response)
            return final_response.content
        else:
            self.messages.append(response)
            return response.content

    def clear_history(self):
        """Clear conversation history (keep system prompt)."""
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]


def build_ask_agent():
    """Build the DefiDoza agent."""
    api_key = os.getenv("GROQ_API_KEY", GROQ_API_KEY)
    if not api_key or api_key == "enter-api-key-here":
        return (
            None,
            "Set GROQ_API_KEY! Replace the placeholder in the script or set as environment variable.",
        )

    try:
        agent = DefiDozaAgent(api_key)
        return agent, None
    except Exception as e:
        return None, f"Failed to create agent: {str(e)}"
