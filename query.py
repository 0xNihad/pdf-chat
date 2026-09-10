from __future__ import annotations

import sys

import config
from embed import co, doc_filter, embed_batch, qdrant
from llm import chat

TOP_K = 5           # excerpts handed to the model
CANDIDATES = 30     # pulled from Qdrant before reranking

SYSTEM = (
    "You answer questions about a document using only the numbered excerpts "
    "given to you. Cite the excerpts you use as [1], [2], etc. If the excerpts "
    "do not contain the answer, say so. Answer in the language of the question. "
    "Write plain prose in short paragraphs. Do not use markdown headings, "
    "bullet characters, or tables."
)


def text_of(hit) -> str:
    return hit.payload.get("contextualized", "")


def retrieve(question: str, doc_id: str, limit: int):
    if not qdrant.collection_exists(config.COLLECTION):
        raise LookupError("that document is no longer available")
    vector = embed_batch([question], "search_query")[0]
    return qdrant.query_points(
        collection_name=config.COLLECTION,
        query=vector,
        query_filter=doc_filter(doc_id),
        limit=limit,
        with_payload=True,
    ).points


def rerank(question: str, hits, top_n: int):
    scorable = [hit for hit in hits if text_of(hit).strip()]
    if not scorable:
        return []
    try:
        response = co.rerank(
            model=config.RERANK_MODEL,
            query=question,
            documents=[text_of(hit) for hit in scorable],
            top_n=min(top_n, len(scorable)),
        )
    except Exception as exc:
        reason = getattr(exc, "body", None) or exc
        if isinstance(reason, dict):
            reason = reason.get("message", reason)
        print(f"warning: rerank unavailable ({reason}); using vector order", file=sys.stderr)
        return [(hit, None) for hit in scorable[:top_n]]
    return [(scorable[result.index], result.relevance_score) for result in response.results]


NO_MATCH = "Nothing in this document matches that question."


def page_of(hit) -> int | None:
    pages = [
        prov["page_no"]
        for item in hit.payload.get("meta", {}).get("doc_items", [])
        for prov in item.get("prov", [])
        if "page_no" in prov
    ]
    return min(pages) if pages else None


MAX_CANDIDATES = 120
HISTORY_TURNS = 3


def candidate_count(chunks: int | None) -> int:
    if not chunks:
        return CANDIDATES
    return max(CANDIDATES, min(MAX_CANDIDATES, chunks // 4))


def answer_question(
    question: str,
    doc_id: str,
    use_rerank: bool = True,
    chunks: int | None = None,
    history: list[dict] | None = None,
) -> dict:
    limit = candidate_count(chunks) if use_rerank else TOP_K
    hits = retrieve(question, doc_id, limit)
    if not hits:
        return {"answer": NO_MATCH, "excerpts": []}

    ranked = rerank(question, hits, TOP_K) if use_rerank else [(h, None) for h in hits[:TOP_K]]
    excerpts = [
        {
            "rank": rank,
            "text": text_of(hit),
            "page": page_of(hit),
            "vector_score": round(hit.score, 4),
            "rerank_score": None if relevance is None else round(relevance, 4),
        }
        for rank, (hit, relevance) in enumerate(ranked, 1)
    ]

    conversation = ""
    for turn in (history or [])[-HISTORY_TURNS:]:
        conversation += f"Earlier question: {turn['question']}\nYour answer: {turn['answer']}\n\n"

    context = "\n\n".join(f"[{e['rank']}] {e['text']}" for e in excerpts)
    answer = chat(SYSTEM, f"{conversation}Excerpts:\n\n{context}\n\nQuestion: {question}")
    return {"answer": answer, "excerpts": excerpts}


def ask(question: str, doc_id: str, use_rerank: bool = True, show_excerpts: bool = True) -> str:
    result = answer_question(question, doc_id, use_rerank)

    if show_excerpts:
        for excerpt in result["excerpts"]:
            score = f"vector={excerpt['vector_score']:.4f}"
            if excerpt["rerank_score"] is not None:
                score += f"  rerank={excerpt['rerank_score']:.4f}"
            print(f"\n[{excerpt['rank']}] {score}\n" + "-" * 60)
            print(excerpt["text"])
        print()

    return result["answer"]
