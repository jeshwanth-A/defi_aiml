# DeFi Predictor and Analyst

A full-stack AI-powered chatbot for DeFi analysis, built with FastAPI and Next.js. Features real-time crypto price lookup, RAG-based document Q&A, and streaming chat via WebSocket.

## Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | Next.js 16, React 19, TypeScript |
| **Backend** | FastAPI, Python 3.10+ |
| **LLM** | LangChain + Groq (Llama 3.3 70B) |
| **RAG** | SentenceTransformers + FAISS |
| **Data** | CoinGecko API (live crypto prices) |

## Features

- Chat with an AI agent that can look up live cryptocurrency prices
- Upload `.md` / `.txt` files for RAG-based context retrieval
- Streaming responses via WebSocket (or standard HTTP with tool use)
- Adjustable conversation memory window
- Toggle between tool-enabled and streaming modes

## Project Structure

```
.
├── backend/
│   ├── main.py          # FastAPI entrypoint
│   ├── config.py         # Pydantic settings (loads .env)
│   ├── state.py          # In-memory app state
│   ├── ai/
│   │   ├── agent.py      # LangChain agent with Groq LLM
│   │   ├── rag.py        # Embeddings + FAISS vector search
│   │   └── tools.py      # LLM tools (crypto price lookup)
│   ├── ml/
│   │   └── data.py       # CoinGecko API data fetcher
│   └── routes/
│       ├── chat.py       # Chat endpoints + WebSocket
│       └── upload.py     # File upload endpoints
├── frontend/
│   └── src/app/
│       └── page.tsx      # Main chat UI
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+
- A [Groq API key](https://console.groq.com/keys)

### Backend Setup

```bash
cd backend

# Create and activate virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# Run the server
uvicorn main:app --reload --port 8432
```

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Run the dev server
npm run dev
```

The frontend runs at `http://localhost:3000` and the backend at `http://localhost:8432`.

## Environment Variables

| Variable | Description | Required |
|---|---|---|
| `GROQ_API_KEY` | API key from [Groq Console](https://console.groq.com/keys) | Yes |

## License

MIT
