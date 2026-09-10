from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
ENV_PATH = ROOT / ".env"

load_dotenv(ENV_PATH)


def _required(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise SystemExit(f"error: {name} is not set - add it to {ENV_PATH}")
    return value


OUTPUT_DIR = (ROOT / os.getenv("OUTPUT_DIR", "output")).resolve()

COHERE_API_KEY = _required("COHERE_API_KEY")
EMBED_MODEL = os.getenv("COHERE_MODEL", "embed-v4.0")
RERANK_MODEL = os.getenv("COHERE_RERANK_MODEL", "rerank-v3.5")

EMBED_DIM = int(os.getenv("EMBED_DIM", "1536"))

MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1024"))

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("QDRANT_COLLECTION", "chatpdf")

LLM_MODEL = os.getenv("LLM_MODEL", "cohere/command-a-03-2025")

OCR_LANG = os.getenv("OCR_LANG", "latin")
