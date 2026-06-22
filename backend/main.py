import logging
import time
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from state import AppState
from config import settings

app = FastAPI()
app.state.data = AppState()

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

logger = logging.getLogger("defi-ai-backend")

from routes import chat_router
from routes import upload_router

app.include_router(chat_router)
app.include_router(upload_router)

@app.get("/")
def root():
    return {"message":"backend is running"}

@app.get("/health")
def health():
    return {
        "status":"ok",
        "service":"defi-ai-backend",
        "version":"1.0.0"
    }

@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    start_time = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception:
        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.exception(
            "%s %s -> 500 %.2fms",
            request.method,
            request.url.path,
            duration_ms
        )
        raise

    duration_ms = (time.perf_counter() - start_time) * 1000

    logger.info(
        "%s %s -> %s %.2fms",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms
    )

    return response

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception(
        "Unhandled backend error on %s %s",
        request.method,
        request.url.path
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": "Unexpected backend error"
        }
    )
