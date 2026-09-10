from __future__ import annotations

import json
import urllib.request
from pathlib import Path

from docling.chunking import HybridChunker
from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
    ChunkingSerializerProvider,
)
from docling_core.transforms.serializer.markdown import (
    MarkdownParams,
    MarkdownTableSerializer,
)
from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer
from docling_core.transforms.serializer.markdown import MarkdownParams
from docling_core.types.doc import DoclingDocument, ImageRefMode
from pydantic import ConfigDict
from tokenizers import Tokenizer

import config
from extract import LAYERS

CACHE_DIR = Path.home() / ".cache" / "cohere_tokenizers"
MODELS_ENDPOINT = "https://api.cohere.com/v1/models"
FALLBACK_URL = "https://storage.googleapis.com/cohere-public/tokenizers/{model}.json"


def _load_vocabulary(model: str, api_key: str) -> Tokenizer:
    """Fetch `model`'s vocabulary once, then read it from disk forever after."""
    cached = CACHE_DIR / f"{model}.json"
    if not cached.is_file():
        url = FALLBACK_URL.format(model=model)
        request = urllib.request.Request(
            f"{MODELS_ENDPOINT}/{model}", headers={"Authorization": f"Bearer {api_key}"}
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                url = json.load(response).get("tokenizer_url") or url
        except Exception:
            pass
        with urllib.request.urlopen(url, timeout=60) as response:
            payload = response.read()
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        partial = cached.with_suffix(".partial")
        partial.write_bytes(payload)
        partial.replace(cached)
    return Tokenizer.from_file(str(cached))


class CohereTokenizer(BaseTokenizer):

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tokenizer: Tokenizer
    max_tokens: int

    def __init__(self, model: str, api_key: str, max_tokens: int):
        super().__init__(
            tokenizer=_load_vocabulary(model, api_key), max_tokens=max_tokens
        )

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False).ids) if text else 0

    def get_max_tokens(self) -> int:
        return self.max_tokens

    def get_tokenizer(self):
        return self.count_tokens


class ChunkSerializerProvider(ChunkingSerializerProvider):

    def get_serializer(self, doc: DoclingDocument) -> ChunkingDocSerializer:
        return ChunkingDocSerializer(
            doc=doc,
            table_serializer=MarkdownTableSerializer(),
            params=MarkdownParams(
                layers=LAYERS,
                image_mode=ImageRefMode.PLACEHOLDER,
                image_placeholder="",
                escape_underscores=False,
                escape_html=False,
                traverse_pictures=True,
            ),
        )


def chunk(doc: DoclingDocument) -> list[dict]:
    chunker = HybridChunker(
        tokenizer=CohereTokenizer(
            model=config.EMBED_MODEL,
            api_key=config.COHERE_API_KEY,
            max_tokens=config.MAX_TOKENS,
        ),
        serializer_provider=ChunkSerializerProvider(),
    )

    records = []
    for index, piece in enumerate(chunker.chunk(doc)):
        record = piece.export_json_dict()
        record["chunk_id"] = index
        record["contextualized"] = chunker.contextualize(piece)
        records.append(record)
    return records
