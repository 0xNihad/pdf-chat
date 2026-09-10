from __future__ import annotations

import time
import uuid

import cohere
from qdrant_client import QdrantClient, models

import config

BATCH_SIZE = 96
RETRY_WAIT = 20
MAX_RETRIES = 5

co = cohere.ClientV2(api_key=config.COHERE_API_KEY)
qdrant = QdrantClient(url=config.QDRANT_URL)


def doc_filter(doc_id: str) -> models.Filter:
    if not doc_id:
        raise ValueError("doc_id is required")
    return models.Filter(
        must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
    )


def embed_batch(texts: list[str], input_type: str) -> list[list[float]]:
    """Embed one batch, retrying a rate-limit rejection a bounded number of times."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return co.embed(
                model=config.EMBED_MODEL,
                texts=texts,
                input_type=input_type,
                embedding_types=["float"],
                output_dimension=config.EMBED_DIM,
            ).embeddings.float_
        except cohere.errors.TooManyRequestsError:
            print(f"  rate-limited; waiting {RETRY_WAIT}s ({attempt}/{MAX_RETRIES})")
            time.sleep(RETRY_WAIT)
    raise RuntimeError(f"still rate-limited after {MAX_RETRIES} retries")


def ensure_collection() -> None:
    if not qdrant.collection_exists(config.COLLECTION):
        qdrant.create_collection(
            collection_name=config.COLLECTION,
            vectors_config=models.VectorParams(
                size=config.EMBED_DIM, distance=models.Distance.COSINE
            ),
        )
        print(f"Created collection '{config.COLLECTION}' (dim={config.EMBED_DIM})")

    qdrant.create_payload_index(
        collection_name=config.COLLECTION,
        field_name="doc_id",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )


def delete_document(doc_id: str) -> None:
    """Remove every point belonging to `doc_id`."""
    if not qdrant.collection_exists(config.COLLECTION):
        return
    qdrant.delete(
        collection_name=config.COLLECTION,
        points_selector=models.FilterSelector(filter=doc_filter(doc_id)),
    )


def index(records: list[dict], doc_id: str) -> int:
    """Embed and store every record under `doc_id`. All or nothing."""
    if not doc_id:
        raise ValueError("doc_id is required")

    ensure_collection()
    delete_document(doc_id)

    started = time.monotonic()
    try:
        for start in range(0, len(records), BATCH_SIZE):
            batch = records[start : start + BATCH_SIZE]
            vectors = embed_batch([r["contextualized"] for r in batch], "search_document")
            qdrant.upsert(
                collection_name=config.COLLECTION,
                points=[
                    models.PointStruct(
                        id=str(uuid.uuid4()),
                        vector=v,
                        payload={**r, "doc_id": doc_id},
                    )
                    for r, v in zip(batch, vectors)
                ],
            )
            print(f"  embedded {min(start + BATCH_SIZE, len(records))}/{len(records)}")
    except BaseException:
        # Half a document is worse than none: remove whatever made it in.
        delete_document(doc_id)
        raise

    stored = qdrant.count(
        config.COLLECTION, count_filter=doc_filter(doc_id), exact=True
    ).count
    print(f"{stored} points for this document ({time.monotonic() - started:.1f}s)")
    return stored