"""Ask questions about a PDF from the command line."""
from __future__ import annotations

import argparse
import json
import sys

import config

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def ingest(args) -> None:
    from chunker import chunk
    from embed import index
    from extract import extract, parse_pages, stable_doc_id
    try:
        doc = extract(args.source, describe_pictures=False, pages=parse_pages(args.pages) if args.pages else None)
    except (FileNotFoundError, RuntimeError) as exc:
        raise SystemExit(f"error: {exc}")

    doc_id = stable_doc_id(doc)
    records = chunk(doc)
    index(records, doc_id)
    chunks_path = config.OUTPUT_DIR / f"{doc_id}_chunks.json"
    chunks_path.write_text(
        json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"{len(records)} chunks -> {chunks_path}")
    print(f"ask it with: python main.py ask \"...\" --doc {doc_id}")


def ask(args) -> None:
    from query import ask as answer

    print(f"\n{args.question}\n" + "=" * 60)
    try:
        print(answer(args.question, args.doc))
    except (LookupError, RuntimeError) as exc:
        raise SystemExit(f"error: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    ingest_cmd = commands.add_parser("ingest", help="convert a PDF and index it")
    ingest_cmd.add_argument("source", help="PDF path or URL")
    ingest_cmd.add_argument("--pages", help="1-based inclusive range, e.g. 1-10")
    # ingest_cmd.add_argument(
    #     "--pictures",
    #     action="store_true",
    #     help="classify and caption pictures (slow; downloads a vision model)",
    # )
    ingest_cmd.set_defaults(run=ingest)

    ask_cmd = commands.add_parser("ask", help="answer a question about one document")
    ask_cmd.add_argument("question")
    ask_cmd.add_argument(
        "--doc",
        required=True,
        help="which document to search: the doc_id printed by `ingest`",
    )
    # ask_cmd.add_argument("--no-rerank", action="store_true", help="skip reranking")
    # ask_cmd.add_argument("--quiet", action="store_true", help="answer only, no excerpts")
    ask_cmd.set_defaults(run=ask)

    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()