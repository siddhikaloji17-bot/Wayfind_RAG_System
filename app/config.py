import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")


class Settings:
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    USE_LOCAL_EMBEDDINGS: bool = os.getenv("USE_LOCAL_EMBEDDINGS", "true").lower() in ("1", "true", "yes")
    LOCAL_EMBEDDING_MODEL: str = os.getenv("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    VECTOR_INDEX_DIR: str = os.getenv("VECTOR_INDEX_DIR", "data/faiss_index")
    HYBRID_TOP_K: int = int(os.getenv("HYBRID_TOP_K", "8"))
    RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "cross-encoder/stsb-roberta-large")
    RERANK_TOP_K: int = int(os.getenv("RERANK_TOP_K", "10"))
    USE_LOCAL_LLM: bool = os.getenv("USE_LOCAL_LLM", "true").lower() in ("1", "true", "yes")
    LOCAL_LLM_MODEL: str = os.getenv("LOCAL_LLM_MODEL", "distilgpt2")
    LOCAL_LLM_MAX_NEW_TOKENS: int = int(os.getenv("LOCAL_LLM_MAX_NEW_TOKENS", "256"))
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))
    GOOGLE_FACTCHECK_API_KEY: str = os.getenv("GOOGLE_FACTCHECK_API_KEY", "")
    GOOGLE_FACTCHECK_ENDPOINT: str = os.getenv("GOOGLE_FACTCHECK_ENDPOINT", "https://factchecktools.googleapis.com/v1alpha1/claims:search")


settings = Settings()
