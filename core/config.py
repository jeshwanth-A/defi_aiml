import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))

IS_COLAB = "google.colab" in sys.modules

if IS_COLAB:
    from google.colab import drive

    drive.mount("/content/drive")
    DEFAULT_DRIVE_BASE = "/content/drive/MyDrive/defidoza"
else:
    DEFAULT_DRIVE_BASE = None

DRIVE_BASE = os.getenv("DRIVE_BASE", DEFAULT_DRIVE_BASE)

if DRIVE_BASE:
    WEIGHTS_DIR = os.path.join(DRIVE_BASE, "weights")
    CACHE_DB = os.path.join(DRIVE_BASE, "cache.db")
    KNOWLEDGE_DIR = os.path.join(DRIVE_BASE, "knowledge")
    RAG_DIR = os.path.join(DRIVE_BASE, "rag")
else:
    WEIGHTS_DIR = os.getenv("WEIGHTS_DIR", "./data/weights")
    CACHE_DB = os.getenv("CACHE_DB", "./data/cache.db")
    KNOWLEDGE_DIR = os.getenv("KNOWLEDGE_DIR", os.path.join(BASE_DIR, "knowledge"))
    RAG_DIR = os.getenv("RAG_DIR", "./data/rag")

WEIGHTS_DIR = os.getenv("WEIGHTS_DIR", WEIGHTS_DIR)
CACHE_DB = os.getenv("CACHE_DB", CACHE_DB)
KNOWLEDGE_DIR = os.getenv("KNOWLEDGE_DIR", KNOWLEDGE_DIR)
RAG_DIR = os.getenv("RAG_DIR", RAG_DIR)

os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(RAG_DIR, exist_ok=True)

SCALER_PATH = os.path.join(WEIGHTS_DIR, "scaler.pkl")
PYTORCH_WEIGHTS = os.path.join(WEIGHTS_DIR, "pytorch_lstm.pth")
TF_WEIGHTS = os.path.join(WEIGHTS_DIR, "tf_lstm.h5")
RF_WEIGHTS = os.path.join(WEIGHTS_DIR, "rf_model.pkl")
RAG_INDEX_PATH = os.path.join(RAG_DIR, "knowledge_index.pkl")
RAG_MANIFEST_PATH = os.path.join(RAG_DIR, "knowledge_manifest.json")

TOKENS = ["uniswap", "bitcoin", "ethereum", "solana", "cardano"]

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_TEMPERATURE = float(os.getenv("GROQ_TEMPERATURE", "0.2"))
RAG_EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "tfidf")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "3"))

_admin_logger = None


def set_admin_logger(logger):
    global _admin_logger
    _admin_logger = logger


def admin_log(category, msg):
    formatted = f"[{category}] {msg}"
    if _admin_logger:
        _admin_logger(formatted)


def get_weights_dir():
    return WEIGHTS_DIR


def get_cache_db():
    return CACHE_DB


def get_knowledge_dir():
    return KNOWLEDGE_DIR


def get_rag_dir():
    return RAG_DIR


def is_colab():
    return IS_COLAB
