# DefiDoza - Crypto Forecasting + Grounded Knowledge Assistant

Multi-model cryptocurrency forecasting with a local tool-calling assistant and a lightweight RAG layer for curated documentation.

[![Forecast Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jeshwanth-A/defi_aiml/blob/main/notebooks/train.ipynb)
[![RAG Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jeshwanth-A/defi_aiml/blob/main/notebooks/rag_prepare.ipynb)

## Overview

- Train forecasting models in Colab and save weights to Google Drive.
- Build a knowledge index in Colab from curated Markdown docs and save it to Google Drive.
- Run one local assistant that can answer both:
  - live price and forecast questions
  - grounded knowledge questions about supported tokens and project notes

## Workflow

1. **Train forecasting models in Colab**
   - Open `notebooks/train.ipynb`
   - Train price models
   - Artifacts are written to `/content/drive/MyDrive/defidoza/weights/`

2. **Prepare the RAG knowledge base in Colab**
   - Open `notebooks/rag_prepare.ipynb`
   - Add or edit Markdown docs in `/content/drive/MyDrive/defidoza/knowledge/`
   - Build the knowledge index
   - Artifacts are written to `/content/drive/MyDrive/defidoza/rag/`

3. **Use the assistant locally**
   - Set `DRIVE_BASE` in `.env`
   - Run `python main.py`
   - Ask for prices, predictions, risks, explanations, or mixed questions

## Local Setup

```bash
git clone https://github.com/jeshwanth-A/defi_aiml.git
cd defi_aiml
pip install -r requirements.txt
```

Create `.env` from the template:

```bash
cp .env.example .env
```

Set at least:

```bash
GROQ_API_KEY=your_key_here
DRIVE_BASE=G:\My Drive\defidoza
```

Optional RAG settings:

```bash
RAG_EMBEDDING_MODEL=tfidf
RAG_TOP_K=3
```

If you want to use the repo's bundled starter docs locally instead of Google Drive, also set:

```bash
KNOWLEDGE_DIR=knowledge
RAG_DIR=data/rag
```

Run:

```bash
python main.py
```

## Knowledge Corpus

Starter curated docs live in `knowledge/` for local development. The RAG notebook can also scaffold a starter corpus directly into Google Drive if `DRIVE_BASE/knowledge/` is empty.

Recommended v1 document types:

- token summaries
- protocol overviews
- risk notes
- glossary entries
- project-specific forecasting notes

## Example Questions

- `What is Solana and what are its main risks?`
- `What does our knowledge base say about Uniswap, and what is its current price?`
- `Give me the forecast for ETH and explain it using our project notes.`

## Project Structure

```text
defi_aiml/
├── core/
│   ├── agent.py          # tool-calling assistant with RAG context injection
│   ├── cache.py          # SQLite cache helpers
│   ├── config.py         # env + path resolution
│   ├── data.py           # preprocessing + sequence builders
│   ├── inference.py      # local prediction logic
│   ├── models.py         # model definitions + weight loading
│   ├── rag.py            # knowledge chunking, indexing, retrieval
│   └── tools.py          # assistant tools
├── knowledge/            # starter curated markdown docs
├── notebooks/
│   ├── train.ipynb       # forecasting model training notebook
│   └── rag_prepare.ipynb # knowledge indexing notebook
├── main.py               # local CLI entry point
├── .env.example
└── requirements.txt
```

## Security / Public Repo Notes

- `.env` is gitignored.
- Model artifacts, vector indexes, and cache files are gitignored through `data/`.
- Do not commit API keys.

## Supported Tokens

Bitcoin, Ethereum, Uniswap, Solana, Cardano.

## License

MIT
