from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from state import AppState
from config import settings

app = FastAPI()
app.state.data = AppState()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_url,settings.frontend_url2],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

from routes import chat_router
from routes import upload_router

app.include_router(chat_router)
app.include_router(upload_router)

@app.get("/")
def a():
    return {"message":"backend is running"}


