import json
import os
from typing import Any

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from pydantic import SecretStr

from core.config import GROQ_API_KEY, GROQ_MODEL, GROQ_TEMPERATURE
from core.rag import format_knowledge_context, get_knowledge_status
from core.tools import ALL_TOOLS, TOOL_MAP


SYSTEM_PROMPT = """You are Doza, a helpful crypto forecasting assistant.

You have access to these tools:
- knowledge_base_status: See whether the local knowledge base is ready
- search_knowledge_base: Search curated docs for grounded explanations and risks
- list_supported_tokens: Get the list of tokens you can work with
- check_model_status: See which ML models are trained
- get_current_price: Get current USD price for a token
- get_price_history: Get recent price history
- predict_price: Run ML prediction for a token's future price

Guidelines:
- Use the provided knowledge base context when it is relevant.
- When users ask explanatory questions, risk summaries, project notes, glossaries, or "what does our knowledge base say", use search_knowledge_base.
- When users ask about prices, use get_current_price.
- When users ask for predictions or forecasts, use predict_price.
- If users ask to compare forecast vs today/current, call both predict_price and get_current_price.
- When users ask about trends or history, use get_price_history.
- When users ask what tokens are supported, use list_supported_tokens.
- When users ask about trained models, use check_model_status.
- Blend knowledge-base results with live tool outputs when the question needs both.
- If the knowledge base is unavailable or has no relevant result, say so plainly instead of inventing project-specific facts.
- If prediction fails because no models are trained, tell the user to use the Train button first.
- Be concise and helpful in your responses.
- Format prices nicely (for example, "$42,123.45")."""


class DefiDozaAgent:
    """Tool-calling agent with grounded knowledge retrieval."""

    def __init__(self, api_key: str):
        self.llm = ChatGroq(
            model=GROQ_MODEL,
            api_key=SecretStr(api_key),
            temperature=GROQ_TEMPERATURE,
        )
        self.llm_with_tools = self.llm.bind_tools(ALL_TOOLS)
        self.messages: list[Any] = [SystemMessage(content=SYSTEM_PROMPT)]

    def _extract_text(self, content: Any) -> str:
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

    def _fallback_from_tool_results(self, working_messages: list[Any]) -> str:
        for msg in reversed(working_messages):
            if not isinstance(msg, ToolMessage):
                continue
            raw_content = msg.content
            if not isinstance(raw_content, (str, bytes, bytearray)):
                continue
            try:
                payload = json.loads(raw_content)
            except Exception:
                continue

            if isinstance(payload, dict) and payload.get("price_usd") is not None:
                token = payload.get("token", "token")
                price = float(payload["price_usd"])
                return f"Latest {token} price is ${price:,.2f}."

            if isinstance(payload, dict) and isinstance(payload.get("results"), list):
                if payload.get("query") is not None:
                    if payload["results"]:
                        top = payload["results"][0]
                        excerpt = str(top.get("content", "")).strip()[:240]
                        return (
                            f"Knowledge base match from {top.get('source', 'document')}: "
                            f"{excerpt}"
                        )
                    return "I checked the knowledge base but did not find a strong match."

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

            if isinstance(payload, dict) and payload.get("error"):
                return f"I hit an error: {payload['error']}"

        return "I ran the request, but could not format the final response. Please try again."

    def _execute_tool(self, tool_call: dict[str, Any]) -> str:
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
            return TOOL_MAP[tool_name].invoke(tool_args)
        except Exception as e:
            return json.dumps({"error": str(e)})

    def _build_knowledge_message(self, question: str) -> SystemMessage | None:
        status = get_knowledge_status()
        if not status.get("index_ready"):
            return None

        context = format_knowledge_context(question)
        if not context:
            return None

        return SystemMessage(
            content=(
                "Knowledge Base Context:\n"
                "Use this retrieved material when it is relevant to the user's question.\n\n"
                f"{context}"
            )
        )

    def ask(self, question: str) -> str:
        """Process a user question using retrieval and tools when helpful."""
        working_messages = list(self.messages)
        knowledge_message = self._build_knowledge_message(question)
        if knowledge_message is not None:
            working_messages.append(knowledge_message)

        working_messages.append(HumanMessage(content=question))
        response = self.llm_with_tools.invoke(working_messages)
        max_rounds = 6

        for _ in range(max_rounds):
            tool_calls = getattr(response, "tool_calls", None)
            if not tool_calls:
                working_messages.append(response)
                self.messages = [
                    msg
                    for msg in working_messages
                    if knowledge_message is None or msg is not knowledge_message
                ]
                text = self._extract_text(response.content)
                return text if text else self._fallback_from_tool_results(working_messages)

            working_messages.append(response)
            for tool_call in tool_calls:
                tool_result = self._execute_tool(tool_call)
                working_messages.append(
                    ToolMessage(content=tool_result, tool_call_id=tool_call["id"])
                )

            response = self.llm_with_tools.invoke(working_messages)

        working_messages.append(response)
        self.messages = [
            msg
            for msg in working_messages
            if knowledge_message is None or msg is not knowledge_message
        ]
        text = self._extract_text(getattr(response, "content", ""))
        return text if text else self._fallback_from_tool_results(working_messages)

    def clear_history(self):
        """Clear conversation history (keep system prompt)."""
        self.messages = [SystemMessage(content=SYSTEM_PROMPT)]


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
