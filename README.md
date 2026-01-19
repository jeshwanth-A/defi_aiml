# DeFi AI/ML: Crypto Price Forecaster

> Multi-model cryptocurrency price prediction with LangChain agent integration and Google Drive persistence

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jeshwanth-A/defi_aiml/blob/main/core.ipynb)

## Features

| Feature | Description |
|---------|-------------|
| **3 ML Models** | PyTorch LSTM, TensorFlow LSTM, Random Forest |
| **LangChain Agent** | Natural language queries with tool execution |
| **Google Drive Storage** | Persistent weights and cache across sessions |
| **Smart Caching** | 24-hour cache for price data and predictions |
| **Admin Logging** | Detailed cache/API/model operation logs |
| **Interactive UI** | ipywidgets dashboard with collapsible panels |

## Quick Start

### 1. Open in Google Colab

Click the badge above or [open directly](https://colab.research.google.com/github/jeshwanth-A/defi_aiml/blob/main/core.ipynb)

### 2. Run the notebook

The notebook will:
- Mount Google Drive (one-time auth per session)
- Create `My Drive/defidoza/` folder for persistence
- Display the interactive UI

### 3. Train Models

1. Click **Train**
2. Select token (bitcoin, ethereum, uniswap, solana, cardano)
3. Choose days of historical data (default: 30)
4. Select model or "All"
5. Click **Start**

### 4. Make Predictions

1. Click **Predict**
2. Select token, days, target date, and model
3. Click **Run**
4. View results with MAE comparison

### 5. Ask Questions (LangChain)

Requires: `pip install langchain langchain-google-genai`

1. Click **Ask**
2. Enter Google API Key (or set `GOOGLE_API_KEY` env var)
3. Ask natural language questions like:
   - "What's the predicted price of bitcoin for tomorrow?"
   - "Compare all models for ethereum"
   - "What's the current price of solana?"

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     UI (ipywidgets)                     │
├──────────┬──────────┬──────────┬───────────────────────┤
│  Train   │ Predict  │   Ask    │   Logs (Toggle)       │
├──────────┴──────────┴──────────┴───────────────────────┤
│                   Core Functions                        │
├─────────────────┬─────────────────┬────────────────────┤
│   Data Layer    │   Model Layer   │   Agent Layer      │
│  - CoinGecko    │  - PyTorch LSTM │  - LangChain       │
│  - SQLite Cache │  - TF LSTM      │  - Gemini LLM      │
│  - Smart Fetch  │  - RandomForest │  - Custom Tools    │
├─────────────────┴─────────────────┴────────────────────┤
│              Google Drive Storage                       │
│  /content/drive/MyDrive/defidoza/                      │
│  ├── cache.db          (price data + predictions)      │
│  └── weights/                                          │
│      ├── pytorch_lstm.pth                              │
│      ├── tf_lstm.h5                                    │
│      ├── rf_model.pkl                                  │
│      └── scaler.pkl                                    │
└─────────────────────────────────────────────────────────┘
```

## UI Layout

```
┌─────────────────────────────────────────┐
│ Trained: pytorch, tensorflow | LangChain OK
│ [Predict] [Ask] [Train]                 │
├─────────────────────────────────────────┤
│ LOG:        [Show] [Clear]              │
│ ┌─────────────────────────────────────┐ │
│ │ Model        Predicted   Actual  MAE│ │
│ │ pytorch      $5.12       $5.36   0.24│
│ └─────────────────────────────────────┘ │
├─────────────────────────────────────────┤
│ ADMIN LOG:  [Show] [Clear]              │
│ ┌─────────────────────────────────────┐ │
│ │ [CACHE] Price data: FOUND (2.5h)   │ │
│ │ [MODEL] Loading pytorch...          │ │
│ │ [MODEL] Inference: 0.21 -> $5.12   │ │
│ └─────────────────────────────────────┘ │
└─────────────────────────────────────────┘
```

## Caching System

### Price Data Cache
- **Key**: `(token, days)`
- **Expiry**: 24 hours
- **Storage**: SQLite `price_data` table

### Prediction Cache
- **Key**: `(token, model, days, target_date)`
- **Expiry**: 24 hours
- **Storage**: SQLite `predictions` table

### Admin Log Categories
| Tag | Description |
|-----|-------------|
| `[CACHE]` | Cache hit/miss/expired status |
| `[API]` | CoinGecko API fetch operations |
| `[SAVE]` | Data/prediction saves |
| `[MODEL]` | Model loading and inference |

## LangChain Agent Tools

| Tool | Description |
|------|-------------|
| `predict_price` | Run ML prediction for a token |
| `get_available_models` | List trained models |
| `get_supported_tokens` | List supported cryptocurrencies |
| `get_current_price` | Get live price from CoinGecko |

### Example Agent Query

```
User: "What's the predicted price of bitcoin for tomorrow using pytorch?"

Agent:
1. Calls get_available_models() → ["pytorch", "tensorflow"]
2. Calls predict_price(token="bitcoin", model="pytorch", target_date="2026-01-20")
3. Returns: "The PyTorch model predicts Bitcoin at $42,150 for tomorrow."
```

## Supported Tokens

- Bitcoin (`bitcoin`)
- Ethereum (`ethereum`)
- Uniswap (`uniswap`)
- Solana (`solana`)
- Cardano (`cardano`)

## Requirements

### Core (auto-installed in Colab)
- PyTorch
- TensorFlow
- scikit-learn
- pandas, numpy
- ipywidgets

### Optional (for Ask feature)
```bash
pip install langchain langchain-google-genai
```

## File Structure

```
defi_aiml/
├── core.ipynb          # Main notebook (run this)
├── Defi_Aiml.ipynb     # Legacy notebook
├── README.md           # This file
└── (on Google Drive)
    └── defidoza/
        ├── cache.db
        └── weights/
```

## Session Persistence

| Item | Persists? | Location |
|------|-----------|----------|
| Model weights | ✅ Yes | Google Drive |
| Price data cache | ✅ Yes | Google Drive |
| Prediction cache | ✅ Yes | Google Drive |
| Scaler | ✅ Yes | Google Drive |

**Train once → Use forever** (until you want to retrain)

## Performance

| Model | Typical MAE | Training Time |
|-------|-------------|---------------|
| PyTorch LSTM | ~$0.20-0.50 | ~30s (50 epochs) |
| TensorFlow LSTM | ~$0.25-0.60 | ~45s (50 epochs) |
| RandomForest | ~$0.30-0.70 | ~5s |

*MAE varies based on token volatility and data range*

## Roadmap

- [x] Multi-model training and prediction
- [x] Google Drive persistence
- [x] LangChain agent integration
- [x] Admin logging system
- [x] Collapsible UI panels
- [ ] More tokens support
- [ ] Backtesting engine
- [ ] Portfolio analysis
- [ ] Real-time streaming

## Author

**Jeshwanth Anumala**
- GitHub: [@jeshwanth-A](https://github.com/jeshwanth-A)
- Email: jeshwanthanumala@gmail.com
- Portfolio: [jeshwanth55.notion.site](https://jeshwanth55.notion.site/portfolio)

## License

MIT License - see [LICENSE](LICENSE) for details.

---

⭐ **Star this repo if you find it useful!**

*Last updated: January 2026*
