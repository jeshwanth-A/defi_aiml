#!/usr/bin/env python3
"""DefiDoza CLI - Command-line chat interface for the crypto agent."""

import os
import sys
import warnings

warnings.filterwarnings(
    "ignore",
    message=r".*google\.api_core.*Python version.*",
    category=FutureWarning,
)

try:
    from sklearn.exceptions import InconsistentVersionWarning

    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except Exception:
    pass

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.agent import build_ask_agent
from core.models import get_trained_models
from core.config import is_colab, get_weights_dir


def main():
    print("=" * 50)
    print("DefiDoza - Crypto Forecasting Agent")
    print("=" * 50)
    print(f"Environment: {'Colab' if is_colab() else 'Local'}")
    print(f"Weights dir: {get_weights_dir()}")
    print()

    trained = get_trained_models()
    if trained:
        print(f"Trained models: {', '.join(trained)}")
    else:
        print("No trained models found. Train models first in the notebook.")
    print()

    print("Building agent...")
    agent, err = build_ask_agent()
    if not agent:
        print(f"Error: {err}")
        print("\nTo fix:")
        print("  1. Set GROQ_API_KEY in your .env file, or")
        print("  2. Export GROQ_API_KEY=your-key-here")
        sys.exit(1)

    print("Agent ready!")
    print("=" * 50)
    print("Commands:")
    print("  - Type your question and press Enter")
    print("  - Type 'quit' or 'exit' to end the session")
    print("  - Type 'clear' to start a new conversation")
    print("=" * 50)
    print()

    while True:
        try:
            user_input = input("You: ").strip()
        except EOFError:
            break

        if not user_input:
            continue

        if user_input.lower() in ["quit", "exit"]:
            print("Goodbye!")
            break

        if user_input.lower() == "clear":
            agent.clear_history()
            print("Conversation cleared.\n")
            continue

        try:
            response = agent.ask(user_input)
            if not response or not str(response).strip():
                response = (
                    "I processed that request, but generated an empty reply. "
                    "Please try once more."
                )
            print(f"\nDefiDoza: {response}\n")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()
