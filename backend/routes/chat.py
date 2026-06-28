from ai import rag_search, lang_ai
from ai.fallback import fallback_response
from fastapi import HTTPException, WebSocket
from fastapi.responses import JSONResponse
from fastapi import APIRouter, Request
from pydantic import BaseModel
import logging
from config import settings

logger = logging.getLogger("defi-ai-backend")

async def safe_fallback_response(query: str) -> str:
    try:
        return await fallback_response(query)
    except Exception:
        logger.exception("Fallback response failed")
        return (
            "The hosted AI provider is temporarily unavailable, and the backend fallback "
            "could not complete this request. Try again in a minute or ask for recent "
            "Ethereum price history."
        )

def json_error_response(request: Request, status_code: int, content: dict) -> JSONResponse:
    response = JSONResponse(status_code=status_code, content=content)
    origin = request.headers.get("origin")
    if origin and origin.rstrip("/") in settings.allowed_origins:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Vary"] = "Origin"
    return response

class ChatRequest(BaseModel):
    message: str
    memory_limit : int = 5

router = APIRouter()

@router.post("/reset")
async def reset(request: Request):
    state = request.app.state.data
    state.memory.clear()
    return {"status":"cleared"}

@router.post("/chat")
async def chat(chat_req: ChatRequest, request: Request):
    state = request.app.state.data
    memory = state.memory
    embeddings = state.embeddings
    chunks = state.chunks
    file_context = state.file_context

    query = chat_req.message
    if not query:
        raise HTTPException(status_code=400, detail="query is required")
    rag_context = ""
    memory_limit = chat_req.memory_limit

    if len(memory) > memory_limit:
        gap = len(memory) - memory_limit
        memorylast = memory[gap: ]
    else :
        memorylast = memory

    if embeddings is not None and len(embeddings) != 0 :
        rag_context = await rag_search(query, embeddings, chunks,top_k = 3)

    try:
        response = await lang_ai(query, rag_context, file_context, memorylast , True )
    except Exception:
        logger.exception("AI provider request failed on /chat; using fallback response")
        response = await safe_fallback_response(query)

    memory.append(f"user: {query}, ai: {response}")
    return {"response": response}


@router.websocket("/ws/chat")
async def ws_chat(websocket: WebSocket):
    state = websocket.app.state.data
    memory = state.memory
    embeddings = state.embeddings
    chunks = state.chunks
    file_context = state.file_context

    await websocket.accept()
    dataweb = await websocket.receive_json()
    query = dataweb["message"]
    rag_context = ""
    memory_limit = dataweb.get("memory_limit",5)

    if len(memory) > memory_limit:
        gap = len(memory) - memory_limit
        memorylast = memory[gap: ]
    else :
        memorylast = memory

    if embeddings is not None and len(embeddings) != 0 :
        rag_context = await rag_search(query, embeddings, chunks, top_k=3)

    f_response = ""
    try:
        response = await lang_ai(query, rag_context, file_context, memorylast, False )

        async for c in response:
            await websocket.send_json({"token": c.content})
            f_response = f_response + c.content
    except Exception:
        logger.exception("AI provider request failed on /ws/chat; using fallback response")
        f_response = await safe_fallback_response(query)
        await websocket.send_json({"token": f_response})

    memory.append(f"user: {query}, ai: {f_response}")
    await websocket.send_json({"done": True})
    await websocket.close()
