# DeFi AI Analyst

A full-stack AI assistant for DeFi market analysis. The project combines a FastAPI backend, LangChain/Groq tool-calling, CoinGecko market-data tools, document-upload RAG with FAISS, WebSocket chat streaming, SQLite TTL caching, and an experimental PyTorch LSTM pipeline for Ethereum next-day price prediction.

## Why This Project Exists

This project is built as a backend + AI application demo. The goal is not to provide financial advice, but to show how an AI assistant can safely combine:

- live external API tools,
- retrieval-augmented generation,
- streaming chat,
- cached tool responses,
- reproducible backend setup with Docker,
- and a small ML inference workflow.

## Tech Stack

| Layer | Technology |
| --- | --- |
| Frontend | Next.js 16, React 19, TypeScript |
| Backend | FastAPI, Python 3.11, Uvicorn |
| AI orchestration | LangChain, Groq Llama 3.3 70B |
| RAG | SentenceTransformers, FAISS |
| Data | CoinGecko API |
| ML | PyTorch LSTM |
| Storage | SQLite TTL cache for tool responses |
| Runtime | Docker, Docker Compose |

## Core Features

- FastAPI backend with `/chat`, `/ws/chat`, `/ragupload`, `/fileupload`, `/reset`, and `/health` endpoints.
- LangChain tool-calling agent with three crypto tools:
  - exact-date CoinGecko lookup,
  - last-N-days market history,
  - Ethereum next-day LSTM prediction.
- SQLite TTL cache for CoinGecko and prediction tool responses.
- Document-upload RAG using SentenceTransformers embeddings and FAISS vector search.
- WebSocket response streaming for chat without tool calls.
- Request logging middleware that records method, path, status code, and latency.
- Global structured error handler for unexpected backend failures.
- Dockerized backend for reproducible local execution.
- Next.js chat UI with file upload boxes, memory control, tool/streaming mode toggle, and prompt shortcuts.

## Architecture

```text
Next.js frontend
    |
    | HTTP / WebSocket
    v
FastAPI backend
    |
    +-- Chat routes
    |     +-- LangChain + Groq assistant
    |     +-- Tool-calling loop
    |
    +-- Upload routes
    |     +-- Text chunking
    |     +-- SentenceTransformers embeddings
    |     +-- FAISS similarity search
    |
    +-- Crypto tools
    |     +-- CoinGecko exact-date data
    |     +-- CoinGecko historical market data
    |     +-- PyTorch LSTM Ethereum prediction
    |
    +-- SQLite TTL cache
```

## Project Structure

```text
.
|-- backend/
|   |-- main.py                 # FastAPI app, CORS, health, logging middleware
|   |-- config.py               # Pydantic settings loaded from .env
|   |-- state.py                # In-memory app state
|   |-- Dockerfile              # Backend container recipe
|   |-- ai/
|   |   |-- agent.py            # LangChain/Groq tool-calling agent
|   |   |-- cache.py            # SQLite TTL cache
|   |   |-- rag.py              # Embeddings and FAISS search
|   |   `-- tools.py            # CoinGecko and prediction tools
|   |-- ml/
|   |   |-- data.py             # CoinGecko data fetching and date parsing
|   |   |-- model.py            # PyTorch LSTM model definition
|   |   |-- predict.py          # LSTM inference pipeline
|   |   |-- ethereum_lstm.pt    # Saved model checkpoint
|   |   `-- notebooks/
|   |       `-- lstm_pytorch.ipynb
|   `-- routes/
|       |-- chat.py             # HTTP chat and WebSocket chat routes
|       `-- upload.py           # RAG/file upload routes
|-- frontend/
|   `-- src/app/page.tsx        # Main chat interface
|-- docker-compose.yml
`-- README.md
```

## Backend API

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | `/` | Basic backend running message |
| GET | `/health` | Health check for service readiness |
| POST | `/chat` | Tool-enabled chat response |
| WS | `/ws/chat` | Streaming chat response |
| POST | `/reset` | Clear in-memory conversation history |
| POST | `/ragupload` | Upload `.md` / `.txt` files for RAG retrieval |
| POST | `/fileupload` | Upload `.md` / `.txt` files as direct file context |

## Run With Docker

Create `backend/.env`:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Start the backend:

```bash
docker compose up --build
```

Verify the service:

```bash
curl http://localhost:8432/health
```

Expected response:

```json
{"status":"ok","service":"defi-ai-backend","version":"1.0.0"}
```

## Run Locally Without Docker

### Backend

Windows:

```powershell
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
uvicorn main:app --reload --port 8432
```

macOS/Linux:

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn main:app --reload --port 8432
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend: `http://localhost:3000`

Backend: `http://localhost:8432`

## Example Prompts

- `give last 30 days ethereum prices`
- `give ethereum price on 23rd Jan`
- `predict ethereum price`
- Upload a `.md` or `.txt` document, then ask a question about its content.

## ML Notes

The Ethereum prediction feature is intentionally scoped and experimental.

- Token: Ethereum
- Target: next-day price
- Features: price, market cap, and volume
- Lookback window: 7 days
- Evaluation: time-based 80/20 split
- Reported notebook metrics: MAE, RMSE, and normalized MAE

The model output should be treated as a demo prediction, not financial advice.

## Limitations

- The assistant predicts only Ethereum next-day price.
- The RAG store is in-memory and resets when the backend restarts.
- Conversation memory is in-memory and not user-authenticated.
- The backend requires a Groq API key for LLM responses.
- CoinGecko requests can be rate-limited, so tool responses are cached with SQLite TTL caching.

## Resume Summary

Built and containerized a FastAPI backend for a DeFi AI assistant with LangChain/Groq tool-calling, CoinGecko market-data tools, SQLite TTL caching, health checks, request logging middleware, document-upload RAG with FAISS, WebSocket chat flows, and an Ethereum PyTorch LSTM prediction pipeline.

## License

MIT
