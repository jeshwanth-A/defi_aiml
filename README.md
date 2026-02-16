# DefiDoza - Crypto Price Forecaster

Multi-model cryptocurrency forecasting with a local tool-calling agent.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jeshwanth-A/defi_aiml/blob/main/notebooks/train.ipynb)

## Overview

- Train models in Colab with a simple ipywidgets panel.
- Save trained artifacts to Google Drive.
- Run the agent locally against those artifacts.

## Workflow

1. **Train in Colab**
   - Open `notebooks/train.ipynb`
   - Run cells and train via the widget UI
   - Artifacts are written to `/content/drive/MyDrive/defidoza/weights/`

2. **Use agent locally**
   - Set your Drive path in `.env` (`DRIVE_BASE=...`)
   - Run `python main.py`
   - Ask for prices and predictions

## Local Setup

```bash
git clone https://github.com/jeshwanth-A/defi_aiml.git
cd defi_aiml
pip install -r requirements.txt
```

Create `.env` from template:

```bash
cp .env.example .env
```

Set at least:

```bash
GROQ_API_KEY=your_key_here
DRIVE_BASE=G:\My Drive\defidoza
```

Run:

```bash
python main.py
```

## Project Structure

```
defi_aiml/
├── core/
│   ├── config.py      # env + path resolution
│   ├── cache.py       # SQLite cache helpers
│   ├── data.py        # preprocessing + sequence builders
│   ├── models.py      # model definitions + weight loading
│   ├── inference.py   # local prediction logic
│   ├── tools.py       # LangChain tools
│   └── agent.py       # DefiDoza agent
├── notebooks/
│   └── train.ipynb    # self-contained training notebook (ipywidgets)
├── main.py            # local CLI entry point
├── .env.example
└── requirements.txt
```

## Security / Public Repo Notes

- `.env` is gitignored.
- Model artifacts and cache are gitignored.
- Do not commit API keys.

## Supported Tokens

Bitcoin, Ethereum, Uniswap, Solana, Cardano.

## License

MIT
