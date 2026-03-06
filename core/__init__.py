import warnings

warnings.filterwarnings(
    "ignore",
    message=r".*google\.api_core.*Python version.*",
    category=FutureWarning,
)

try:
    from sklearn.exceptions import InconsistentVersionWarning

    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except Exception:
    pass

from core.config import (
    IS_COLAB,
    TOKENS,
    WEIGHTS_DIR,
    CACHE_DB,
    KNOWLEDGE_DIR,
    RAG_DIR,
    GROQ_API_KEY,
    GROQ_MODEL,
    GROQ_TEMPERATURE,
    RAG_EMBEDDING_MODEL,
    RAG_TOP_K,
    admin_log,
    set_admin_logger,
    get_weights_dir,
    get_cache_db,
    get_knowledge_dir,
    get_rag_dir,
    is_colab,
)

from core.cache import (
    init_cache,
    save_price_data,
    get_cached_data,
    get_cached_prediction,
    save_prediction,
)

from core.data import (
    fetch_and_parse,
    smart_fetch,
    preprocess_data,
    create_sequences,
)

from core.models import (
    PricePredictor,
    create_tf_model,
    create_rf_model,
    weights_exist,
    get_trained_models,
    load_model,
)

from core.inference import (
    inverse_transform_price,
    run_inference,
    get_actual_price,
)

from core.rag import (
    build_index_from_directory,
    format_knowledge_context,
    get_knowledge_status,
    load_knowledge_chunks,
    retrieve_knowledge,
)

from core.tools import (
    knowledge_base_status,
    search_knowledge_base,
    list_supported_tokens,
    check_model_status,
    get_current_price,
    get_price_history,
    predict_price,
    ALL_TOOLS,
    TOOL_MAP,
)

from core.agent import (
    DefiDozaAgent,
    build_ask_agent,
)

__version__ = "0.1.0"
__all__ = [
    "IS_COLAB",
    "TOKENS",
    "WEIGHTS_DIR",
    "CACHE_DB",
    "KNOWLEDGE_DIR",
    "RAG_DIR",
    "admin_log",
    "set_admin_logger",
    "get_weights_dir",
    "get_cache_db",
    "get_knowledge_dir",
    "get_rag_dir",
    "is_colab",
    "RAG_EMBEDDING_MODEL",
    "RAG_TOP_K",
    "init_cache",
    "save_price_data",
    "get_cached_data",
    "get_cached_prediction",
    "save_prediction",
    "fetch_and_parse",
    "smart_fetch",
    "preprocess_data",
    "create_sequences",
    "PricePredictor",
    "create_tf_model",
    "create_rf_model",
    "weights_exist",
    "get_trained_models",
    "load_model",
    "inverse_transform_price",
    "run_inference",
    "get_actual_price",
    "build_index_from_directory",
    "format_knowledge_context",
    "get_knowledge_status",
    "load_knowledge_chunks",
    "retrieve_knowledge",
    "knowledge_base_status",
    "search_knowledge_base",
    "list_supported_tokens",
    "check_model_status",
    "get_current_price",
    "get_price_history",
    "predict_price",
    "ALL_TOOLS",
    "TOOL_MAP",
    "DefiDozaAgent",
    "build_ask_agent",
]
