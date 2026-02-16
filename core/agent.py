import json
import os
from typing import Any

from langchain_groq import ChatGroq
from langchain_core.messages import ToolMessage
from pydantic import SecretStr

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
- If users ask to compare forecast vs today/current, call both predict_price and get_current_price
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
            api_key=SecretStr(api_key),
            temperature=GROQ_TEMPERATURE,
        )
        self.llm_with_tools = self.llm.bind_tools(ALL_TOOLS)
        self.messages: list[Any] = [{"role": "system", "content": SYSTEM_PROMPT}]

    def _extract_text(self, content) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            texts = []
            for item in content:
                if isinstance(item, str):
                    texts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text") or item.get("content")
                    if isinstance(text, str):
                        texts.append(text)
            return "\n".join(t for t in texts if t).strip()
        return str(content).strip() if content is not None else ""

    def _fallback_from_tool_results(self) -> str:
        for msg in reversed(self.messages):
            if not isinstance(msg, ToolMessage):
                continue
            raw_content = msg.content
            if not isinstance(raw_content, (str, bytes, bytearray)):
                continue
            try:
                payload = json.loads(raw_content)
            except Exception:
                continue

            if isinstance(payload, dict) and isinstance(payload.get("results"), list):
                token = payload.get("token", "token")
                target_date = payload.get("target_date", "target date")
                parts = []
                for row in payload["results"]:
                    model = row.get("model", "model")
                    if row.get("predicted_usd") is not None:
                        parts.append(f"{model}: ${float(row['predicted_usd']):,.2f}")
                    elif row.get("error"):
                        parts.append(f"{model}: {row['error']}")
                if parts:
                    return f"Prediction for {token} on {target_date} -> " + "; ".join(
                        parts
                    )

            if isinstance(payload, dict) and payload.get("price_usd") is not None:
                token = payload.get("token", "token")
                price = float(payload["price_usd"])
                return f"Latest {token} price is ${price:,.2f}."

            if isinstance(payload, dict) and payload.get("error"):
                return f"I hit an error: {payload['error']}"

        return "I ran the request, but could not format the final response. Please try again."

    def _execute_tool(self, tool_call) -> str:
        """Execute a tool call and return the result."""
        tool_name = tool_call["name"]
        tool_args = tool_call.get("args", {})

        if isinstance(tool_args, (str, bytes, bytearray)):
            try:
                tool_args = json.loads(tool_args)
            except Exception:
                tool_args = {}

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
        max_rounds = 6

        for _ in range(max_rounds):
            tool_calls = getattr(response, "tool_calls", None)
            if not tool_calls:
                self.messages.append(response)
                text = self._extract_text(response.content)
                return text if text else self._fallback_from_tool_results()

            self.messages.append(response)
            for tool_call in tool_calls:
                tool_result = self._execute_tool(tool_call)
                self.messages.append(
                    ToolMessage(content=tool_result, tool_call_id=tool_call["id"])
                )

            response = self.llm_with_tools.invoke(self.messages)

        # Safety net in case the model keeps requesting tools.
        self.messages.append(response)
        text = self._extract_text(getattr(response, "content", ""))
        return text if text else self._fallback_from_tool_results()

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
