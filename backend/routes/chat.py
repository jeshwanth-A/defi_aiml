from ai import rag_search, lang_ai
from fastapi import HTTPException, WebSocket
from fastapi.responses import JSONResponse
from fastapi import APIRouter, Request
from pydantic import BaseModel
import logging

logger = logging.getLogger("defi-ai-backend")

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
        logger.exception("AI provider request failed on /chat")
        return JSONResponse(
            status_code=502,
            content={
                "error": "ai_provider_error",
                "message": "AI provider request failed. Please retry."
            }
        )

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

    response = await lang_ai(query, rag_context, file_context, memorylast, False )

    f_response = ""
    async for c in response:
        await websocket.send_json({"token": c.content})
        f_response = f_response + c.content

    memory.append(f"user: {query}, ai: {f_response}")
    await websocket.send_json({"done": True})
    await websocket.close()
