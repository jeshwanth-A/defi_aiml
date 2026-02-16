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
else:
    WEIGHTS_DIR = os.getenv("WEIGHTS_DIR", "./data/weights")
    CACHE_DB = os.getenv("CACHE_DB", "./data/cache.db")

os.makedirs(WEIGHTS_DIR, exist_ok=True)

SCALER_PATH = os.path.join(WEIGHTS_DIR, "scaler.pkl")
PYTORCH_WEIGHTS = os.path.join(WEIGHTS_DIR, "pytorch_lstm.pth")
TF_WEIGHTS = os.path.join(WEIGHTS_DIR, "tf_lstm.h5")
RF_WEIGHTS = os.path.join(WEIGHTS_DIR, "rf_model.pkl")

TOKENS = ["uniswap", "bitcoin", "ethereum", "solana", "cardano"]

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_TEMPERATURE = float(os.getenv("GROQ_TEMPERATURE", "0.2"))

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


def is_colab():
    return IS_COLAB
