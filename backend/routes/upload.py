from ai import embed, chunky
from fastapi import File, UploadFile, APIRouter, Request
import numpy as np
from config import settings
router = APIRouter()

@router.post("/ragupload")
async def upload_ragfiles(request: Request, files: list[UploadFile] = File(...)):
    state = request.app.state.data
    chunks = state.chunks

    context = ""

    for file in files:
        content = await file.read()
        text = content.decode("utf-8")
        context = context + text

    new_chunks = chunky(context, settings.maxi)
    for i in new_chunks:
        chunks.append(i)

    new_embeddings = embed(new_chunks)
    if state.embeddings is None:
        state.embeddings = new_embeddings
    else:
        state.embeddings = np.vstack([state.embeddings, new_embeddings])

    return {"files": [{"name": f.filename, "size": f.size} for f in files]}

@router.post("/fileupload")
async def upload_files(request: Request, files: list[UploadFile] = File(...)):
    state = request.app.state.data

    for file in files:
        content = await file.read()
        text = content.decode("utf-8")
        state.file_context += text

    return {"files": [{"name": f.filename, "size": f.size} for f in files]}
