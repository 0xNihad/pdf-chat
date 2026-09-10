# pdfextractor

Ask questions about a PDF and get answers that cite the exact passages they came from.

You give it a PDF (a file or a URL). It reads the PDF, splits it into chunks, and stores them in a vector database. Then you ask questions, and it finds the most relevant chunks and has an LLM answer from those chunks only.

```
python main.py ingest report.pdf
python main.py ask "what is the filing deadline?" --doc <doc_id>
```

## How it works

**Ingest** (`python main.py ingest`)

1. **Convert.** [Docling](https://github.com/docling-project/docling) reads the PDF: text, headings, tables, and the reading order. Scanned pages go through OCR (RapidOCR).
2. **Chunk.** The document is split into chunks of up to `MAX_TOKENS` tokens, following its structure, so a chunk doesn't cut across sections. Each chunk gets its section headings in front of it, so a line like "one year from the date of filing" still says what it's about.
3. **Embed and store.** Each chunk is turned into a vector with Cohere and stored in Qdrant, tagged with the document's `doc_id`.

**Ask** (`python main.py ask`)

1. The question is embedded and compared against the chunks of that one document.
2. The best candidates are reranked with Cohere, and the top 5 are kept.
3. The LLM answers from those 5 excerpts only, citing them as [1], [2], and so on. If they don't contain the answer, it says so.

## Requirements

- Python 3.10+
- Docker (for Qdrant)
- A [Cohere API key](https://dashboard.cohere.com/api-keys). It's used for embeddings, reranking, and the tokenizer. A free trial key works, but it's rate-limited.
- An API key for the LLM that writes the answers. Cohere works here too, so one key can cover everything.

## Setup

```
git clone <this repo>
cd pdfextractor

python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # macOS / Linux

pip install docling "docling-core[chunking]" cohere qdrant-client litellm python-dotenv tokenizers

copy .env.example .env           # Windows
# cp .env.example .env           # macOS / Linux
```

Open `.env` and fill in `COHERE_API_KEY`, plus the key for your LLM if it isn't Cohere.

Start Qdrant:

```
docker compose up -d
```

## Usage

### Ingest a PDF

```
python main.py ingest contract.pdf
python main.py ingest "C:\path with spaces\contract.pdf"
python main.py ingest https://arxiv.org/pdf/2408.09869
```

When it finishes, it prints the document's id and the command to ask about it:

```
26 chunks -> output\d39af078-..._chunks.json
ask it with: python main.py ask "..." --doc d39af078-0c26-53d0-8fa1-929f95480bc6
```

The first run is slower, because it downloads the OCR and layout models. Later runs load them from disk.

### Ask a question

```
python main.py ask "what does TableFormer do?" --doc d39af078-0c26-53d0-8fa1-929f95480bc6
```

It prints the 5 excerpts it used, with their scores, then the answer.

### About the doc id

The `doc_id` comes from the PDF's bytes, not its name. That means:

- The same file always gets the same id, so ingesting it again **replaces** the old copy instead of storing it twice.
- Two different files never share an id, even if they have the same name.
- An edited version of a file counts as a new document. The old version stays in Qdrant under its old id.

## Configuration

Everything is set in `.env`. See `.env.example` for the full list.

| Setting | Default | What it does |
|---|---|---|
| `COHERE_API_KEY` | (required) | Embeddings, reranking, and the chunker's tokenizer |
| `COHERE_MODEL` | `embed-v4.0` | Embedding model |
| `COHERE_RERANK_MODEL` | `rerank-v3.5` | Rerank model |
| `EMBED_DIM` | `1536` | Vector size. Changing it means re-ingesting everything. |
| `MAX_TOKENS` | `1024` | Maximum tokens per chunk |
| `LLM_MODEL` | `cohere/command-a-03-2025` | The model that writes answers. Anything [LiteLLM](https://docs.litellm.ai/docs/providers) supports, e.g. `openai/gpt-4o`, `anthropic/claude-opus-5`, `ollama/llama3`. |
| `QDRANT_URL` | `http://localhost:6333` | Where Qdrant runs |
| `QDRANT_COLLECTION` | `pdfextractor` | Collection name |
| `OCR_LANG` | `latin` | OCR script. `latin` covers English, Azerbaijani, and Turkish. Others: `cyrillic`, `arabic`, `devanagari`, `korean`, `japan`, `ch`. |
| `OUTPUT_DIR` | `output` | Where the converted files go |

To use the [OpenCode](https://opencode.ai) gateway, set `LLM_MODEL=opencode/<model>` and `OPENCODE_API_KEY`.

## Output files

For every ingested document, `output/` gets three files, named by its `doc_id`:

| File | What's in it |
|---|---|
| `<doc_id>.json` | The full converted document, in Docling's format |
| `<doc_id>.md` | The same document as readable markdown, useful to check the conversion |
| `<doc_id>_chunks.json` | Every chunk exactly as it was embedded and stored |

## Project layout

| File | Job |
|---|---|
| `main.py` | The CLI: `ingest` and `ask` |
| `extract.py` | PDF → Docling document, and the `doc_id` |
| `chunker.py` | Document → chunks, counted with Cohere's tokenizer |
| `embed.py` | Chunks → Cohere vectors → Qdrant |
| `query.py` | Search, rerank, and build the answer |
| `llm.py` | One call to whichever LLM `.env` points at |
| `config.py` | Reads `.env` |
| `docker-compose.yml` | Qdrant |