# DeFi AI/ML: Crypto Price Forecaster

> Multi-model cryptocurrency price prediction with LangChain agent integration

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jeshwanth-A/defi_aiml/blob/main/core.ipynb)

## Overview

A cryptocurrency price forecasting system that combines multiple ML models with a conversational AI interface. Built for Google Colab with persistent storage via Google Drive.

## Features

- **Multi-Model Predictions** — PyTorch LSTM, TensorFlow LSTM, Random Forest
- **LangChain Agent** — Natural language queries with tool execution
- **Persistent Storage** — Weights and cache saved to Google Drive
- **Smart Caching** — Reduces API calls and computation
- **Interactive UI** — ipywidgets dashboard with collapsible panels
- **Admin Logging** — Detailed operation logs for debugging

## Quick Start

1. Click the **Open in Colab** badge above
2. Run the notebook (authorizes Google Drive on first run)
3. **Train** → Select token and model → Start training
4. **Predict** → Choose parameters → View results
5. **Ask** → Natural language queries (requires API key)

## Supported Tokens

Bitcoin • Ethereum • Uniswap • Solana • Cardano

## Tech Stack

| Category | Technologies |
|----------|--------------|
| ML/DL | PyTorch, TensorFlow, scikit-learn |
| LLM | LangChain, Google Gemini |
| Data | CoinGecko API, SQLite, pandas |
| UI | ipywidgets |
| Storage | Google Drive |

## Project Structure

```
defi_aiml/
├── core.ipynb    # Main notebook (run this)
└── README.md
```

## Requirements

Core dependencies are auto-installed in Colab. For LangChain features:

```bash
pip install langchain langchain-google-genai
```

## Author

**Jeshwanth Anumala**  
[GitHub](https://github.com/jeshwanth-A) • [Portfolio](https://jeshwanth55.notion.site/portfolio) • jeshwanthanumala@gmail.com

## License

MIT License
