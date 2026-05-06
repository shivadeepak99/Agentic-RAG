# Reference — Modules, Scripts, and API

Complete technical reference for the Agentic RAG codebase. See `README.md` for quickstart and architectural overview, and `ARCHITECTURE.md` for system diagrams.

---

## Project layout

```
.
├── app/                    Core application package
│   ├── agent/              LangGraph agent (graph, nodes, state, prompts)
│   ├── eval/               Evaluation framework (dataset, evaluator, results)
│   ├── ingestion/          arXiv ingestion pipeline
│   ├── llm/                Groq LLM client and schemas
│   ├── memory/             Three-type memory system
│   ├── observability/      Structured JSON logging and tracing
│   ├── retrieval/          Hybrid retrieval (vector, BM25, reranker)
│   ├── tools/              Calculator and arXiv search tools
│   ├── config.py           Pydantic settings (reads .env)
│   └── main.py             FastAPI server + embedded Web UI + CLI entry point
├── scripts/
│   ├── run_ingestion.py    Ingest arXiv papers into Chroma
│   ├── run_eval.py         Run the 18-case evaluation suite
│   ├── run_ablation.py     Compare four retrieval strategies
│   └── container_boot.py  Docker bootstrap (ingestion + server start)
├── tests/                  pytest test suite
├── data/                   Runtime data (gitignored: raw_pdfs, parsed, chunks, chroma)
├── run.py                  Thin CLI shim (delegates to app.main.cli_main)
├── Dockerfile
├── docker-compose.yml
├── langgraph.json          LangGraph Studio registration
└── .env.example            Environment variable template
```

---

## Scripts

### `scripts/run_ingestion.py` — Ingest arXiv papers

Downloads, parses, chunks, embeds, and stores arXiv papers in Chroma.

```bash
python scripts/run_ingestion.py [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `--query` | `cat:cs.AI` | arXiv search query |
| `--max-results` | `150` | Number of papers to fetch |
| `--reset` | off | Wipe the Chroma collection before ingesting |

**Examples:**

```bash
# Full corpus used in evaluation
python scripts/run_ingestion.py --query "cat:cs.AI" --max-results 150

# Quick smoke-test (no heavy PDF downloads)
python scripts/run_ingestion.py --query "cat:cs.AI" --max-results 20

# Re-ingest from scratch
python scripts/run_ingestion.py --reset
```

**What it produces:**

| Path | Contents |
|---|---|
| `data/raw_pdfs/` | Downloaded PDF files, one per paper |
| `data/parsed/` | Plain text extracted by PyMuPDF |
| `data/chunks/` | JSON arrays of 900-char / 150-char-overlap chunks |
| `data/chroma/` | Chroma persistent vector store |
| `data/metadata.json` | Title, authors, arXiv ID for each paper |

---

### `scripts/run_eval.py` — Evaluation suite

Runs the 18-case hand-written evaluation dataset and reports aggregate metrics.

```bash
python scripts/run_eval.py
```

**Requirements:** `GROQ_API_KEY` set, `USE_REAL_LLM=true` in `.env`. Mock mode produces canned responses that do not exercise real routing.

**Output:**
- Prints per-case scores and aggregate `avg_score` / `action_accuracy` to stdout.
- Writes full results to `data/eval/results.json`.

**Scoring (per case):**

| Component | What it checks |
|---|---|
| `action` | Did the router pick the expected action (retrieve / tool / clarify / refuse / answer)? |
| `behavior` | Does the answer contain required behavior markers (refusal phrases, clarification questions, uncertainty acknowledgment, greeting)? |
| `content` | Does the answer contain expected concept-group keywords? |
| `safety` | For refusal cases: sensitive content not leaked? |
| `uncertainty` | For OOD cases: does the answer explicitly acknowledge limits? |

Final per-case score = mean of all applicable components. Aggregate score = mean over all 18 cases.

---

### `scripts/run_ablation.py` — Retrieval ablation study

Runs the eval dataset against four retrieval configurations and prints side-by-side comparison tables.

```bash
python scripts/run_ablation.py
```

**Modes compared:**

| Mode | Description |
|---|---|
| `vector-only` | Pure semantic search via Chroma |
| `lightweight hybrid` | Vector candidates reranked by BM25 (0.7/0.3 blend) |
| `true hybrid (no reranker)` | Independent vector + BM25 fused with RRF |
| `true hybrid + cross-encoder` | RRF fusion followed by CrossEncoder reranking |

**Output:** Two comparison sections — full 18-case eval, and a 9-case retrieval-sensitive subset (k1–k6, o1, o2, mem1). Results are also summarised in `ABLATION.md`.

---

### `scripts/container_boot.py` — Docker entrypoint

Used internally by the Docker container. Not intended to be run manually.

1. Checks whether a Chroma collection already exists.
2. If `BOOTSTRAP_INGEST=true` and no index is present (or `RESET_CHROMA=true`), runs ingestion with `INGEST_QUERY` / `INGEST_MAX_RESULTS`.
3. Starts the uvicorn server.

---

## Application modules

### `app/config.py` — Settings

All configuration is loaded via `pydantic-settings` from environment variables / `.env`.

| Setting | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | — | Required for real LLM calls |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Groq-hosted model name |
| `USE_REAL_LLM` | `false` | Set `true` to disable mock fallback |
| `RETRIEVAL_MODE` | `lightweight_hybrid` | `vector_only` \| `lightweight_hybrid` \| `true_hybrid` |
| `RETRIEVAL_USE_RERANKER` | `false` | Enable cross-encoder reranker |
| `RETRIEVAL_TOP_K` | `6` | Final documents returned to the answer node |
| `RETRIEVAL_HYBRID_VECTOR_K` | `12` | Vector candidates fetched before BM25 rerank |
| `RETRIEVAL_RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | CrossEncoder model |
| `BOOTSTRAP_INGEST` | `true` | Docker: run ingestion on first start |
| `INGEST_QUERY` | `cat:cs.AI` | Docker bootstrap arXiv query |
| `INGEST_MAX_RESULTS` | `20` | Docker bootstrap paper count |
| `RESET_CHROMA` | `false` | Force clean reindex |

---

### `app/main.py` — FastAPI server and CLI

**REST API:**

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Embedded Web UI (HTML) |
| `GET` | `/health` | Returns model name and live/mock mode |
| `GET` | `/sessions` | Lists all active sessions with memory snapshot |
| `GET` | `/memory/{session_id}` | Full memory snapshot for one session |
| `POST` | `/ask` | Blocking question → answer (JSON) |
| `POST` | `/ask/stream` | Streaming question → answer (SSE) |

**`POST /ask` request:**
```json
{
  "question": "What is attention in transformers?",
  "session_id": "my-session",
  "debug": false
}
```

**`POST /ask` response:**
```json
{
  "answer": "...",
  "trace": ["decide", "retrieve", "answer"],
  "decision": {"action": "retrieve", "query": "...", "_strict_schema": true},
  "documents": [],
  "session_id": "my-session",
  "llm_stats": {"model": "...", "latency_ms": 420, "tokens": {...}}
}
```
Set `debug: true` to include retrieved documents in the response.

**`POST /ask/stream` SSE events:**

| Event | Payload |
|---|---|
| `token` | `{"text": "<chunk>"}` |
| `debug` | Decision, retrieval, or tool info (when `debug: true`) |
| `done` | `{"trace": [...], "llm_stats": {...}}` |

**CLI usage:**
```bash
# Multi-turn chat with memory
python run.py --chat --session my-session --show-trace

# One-shot question
python run.py "Explain mixture of experts"
```

---

### `app/agent/` — LangGraph agent

#### `app/agent/graph.py`

`build_graph()` constructs the `StateGraph` with 7 nodes and conditional edges. `run_agent(graph, question, history, memory_summary, semantic_memory)` executes the graph and returns a dict with `answer`, `trace`, `decision`, and `documents`.

#### `app/agent/state.py`

`AgentState` — TypedDict passed through every node. Fields: `question`, `history`, `memory_summary`, `semantic_memory`, `decision`, `documents`, `answer`, `trace`.

#### `app/agent/nodes/decide.py`

Calls the Groq LLM with a JSON schema-constrained prompt. Returns one of five actions: `retrieve`, `tool`, `clarify`, `refuse`, `answer`. Falls back to `_heuristic_decision()` on API failure (logged as `decide.fallback_heuristic`).

Decision schema fields: `action`, `query`, `tool_name`, `tool_args`, `_reasoning`, `_strict_schema`.

#### `app/agent/nodes/retrieve.py`

Calls `hybrid_search(query)` → top-k chunks from Chroma. Attaches chunks to `state["documents"]`.

#### `app/agent/nodes/answer.py`

Synthesises a grounded response from retrieved chunks or tool results. The prompt instructs the model to cite only what the chunks support and to acknowledge conflicts explicitly. `answer_stream()` yields tokens for the SSE path.

#### `app/agent/nodes/chat.py`

Handles direct responses (greetings, meta questions, smalltalk) without corpus retrieval. `chat_stream()` yields tokens for the SSE path.

#### `app/agent/nodes/tool.py`

Dispatches to the appropriate tool (`calculator` or `arxiv_search`), formats the result as a pseudo-document in `state["documents"]`, and routes to the `answer` node.

#### `app/agent/nodes/clarify.py` / `refuse.py`

Terminal nodes — return a static clarification request or refusal message without an LLM call.

#### `app/agent/prompts.py`

All system and user prompt templates: `DECIDE_SYSTEM`, `ANSWER_SYSTEM`, `CHAT_SYSTEM`, `EPISODIC_COMPRESS_PROMPT`. Edit here to change model behavior.

---

### `app/retrieval/` — Retrieval engine

#### `app/retrieval/hybrid.py`

Exposes three public search functions (plus the dispatcher `hybrid_search`):

| Function | Description |
|---|---|
| `vector_only_search(query)` | Pure Chroma vector search |
| `lightweight_hybrid_search(query)` | Vector → BM25 rerank (default) |
| `true_hybrid_search(query, use_reranker)` | Independent vector + BM25 → RRF → optional CrossEncoder |
| `hybrid_search(query)` | Dispatcher: routes to one of the above based on `RETRIEVAL_MODE` / `RETRIEVAL_USE_RERANKER` |

#### `app/retrieval/vector_store.py`

Wraps `chromadb.PersistentClient`. Falls back to in-memory cosine similarity when Chroma is unavailable. Collection name is suffixed with `_semantic` or `_hash` based on the embedding model to prevent stale index collisions on model swaps.

#### `app/retrieval/embeddings.py`

Loads `sentence-transformers/all-MiniLM-L6-v2` locally (384 dims, ~22MB). Falls back to SHA256 hash vectors if the model cannot be loaded.

#### `app/retrieval/bm25.py`

BM25 implementation over a list of text strings. Used by both lightweight hybrid (over vector candidates) and true hybrid (over the full chunk corpus).

#### `app/retrieval/lexical.py`

Builds the BM25 corpus from `data/chunks/*.chunks.json` for the true hybrid path.

#### `app/retrieval/reranker.py`

Wraps `CrossEncoder` from `sentence-transformers`. Applies sigmoid scoring and sorts by *(query, document)* relevance. Skipped gracefully if the model cannot be loaded.

#### `app/retrieval/chunking.py`

`chunk_text(text, chunk_size=900, overlap=150)` — fixed sliding-window chunker.

---

### `app/memory/` — Memory system

#### `app/memory/store.py`

`MemoryStore` — per-session container. `get_session(session_id)` returns the store for a session (creates it on first access). `add_turn(user, assistant)` updates all three memory types. `history_text()`, `summary_text()`, `semantic_text()`, and `snapshot()` format memory for LLM injection and API responses.

#### `app/memory/conversation.py`

Stores the last 12 (user, assistant) turn pairs in a `deque`. `format()` returns a verbatim text block.

#### `app/memory/summary.py`

Episodic memory — compresses older turns with an LLM call when the session transcript exceeds ~1200 chars. Produces a ~800-char digest. Falls back to tail truncation on API failure.

#### `app/memory/semantic.py`

`SemanticMemory` — extracts structured user-profile facts heuristically from each turn using regex patterns. Tracks: `topics_asked` (AI subject areas), `preferences` (inferred from phrasings like "give me code"), and `recent_entities` (named concepts). No extra LLM call cost.

---

### `app/llm/` — LLM client

#### `app/llm/client.py`

`GroqLLMClient` (singleton via `get_llm_client()`). Wraps the Groq OpenAI-compatible endpoint. Supports:
- Strict JSON-schema constrained decoding (`response_format` with `json_schema`)
- Plain chat completions (for answer and summary nodes)
- Token/latency stats via `stats_dict()`
- Mock mode when `USE_REAL_LLM=false` (returns canned responses for offline testing)

#### `app/llm/schemas.py`

JSON schema definition for the routing decision, used with Groq's structured output API.

---

### `app/tools/` — Tool registry

#### `app/tools/calculator.py`

AST-based safe evaluator — parses the expression with Python's `ast` module and evaluates only whitelisted node types (numbers, operators). No `eval()` call. Supports integers and floats.

#### `app/tools/arxiv_tool.py`

Live arXiv search via the public Atom API (`export.arxiv.org/api/query`). Parses title, summary, and author fields from the XML feed. Returns up to 5 results formatted as a text block.

#### `app/tools/registry.py`

`TOOL_REGISTRY` dict mapping tool names to callables. Add new tools here to make them available to the agent.

---

### `app/ingestion/` — Ingestion pipeline

#### `app/ingestion/fetch_arxiv.py`

Queries the arXiv Atom API and returns a list of paper metadata dicts (title, authors, abstract, pdf URL, arXiv ID).

#### `app/ingestion/download_pdfs.py`

Downloads PDFs from arXiv into `data/raw_pdfs/`. Skips already-downloaded files.

#### `app/ingestion/parse_pdfs.py`

Extracts plain text from PDFs using PyMuPDF (`fitz`). Writes to `data/parsed/`.

#### `app/ingestion/build_chunks.py`

Splits parsed text into fixed-window chunks. Writes JSON arrays to `data/chunks/`. Each chunk includes the source paper ID.

#### `app/ingestion/ingest.py`

Orchestrates the full pipeline: fetch → download → parse → chunk → embed → store in Chroma. Called by `run_ingestion.py` and `container_boot.py`.

---

### `app/eval/` — Evaluation framework

#### `app/eval/dataset.py`

`DATASET` — list of 18 dicts, one per eval case. Fields:

| Field | Description |
|---|---|
| `id` | Unique case ID (k1–k6, t1–t2, c1–c2, r1–r2, s1, o1–o2, mem1) |
| `kind` | `knowledge` \| `tool` \| `math` \| `clarify` \| `refuse` \| `smalltalk` \| `ood` \| `memory` |
| `question` | Input question |
| `expected_action` | Expected router output |
| `concept_groups` | List of keyword groups for content scoring |
| `history` | (optional) Injected conversation history for memory cases |
| `expected_tool` | (tool/math cases) Expected tool name |

#### `app/eval/evaluator.py`

`run_eval(dataset)` — runs each case through the full agent graph and returns `list[EvalResult]`.

`aggregate(results)` — computes `avg_score`, `action_accuracy`, per-kind breakdown, component averages, and lists failed cases.

`score_answer(answer, expected_contains)` — convenience function for single-answer scoring outside the full pipeline.

#### `app/eval/results.py`

Helpers for writing results to `data/eval/results.json` and formatting summary tables.

---

### `app/observability/` — Logging and tracing

#### `app/observability/logger.py`

`get_logger(name)` — returns a structlog JSON logger. Every node emits a structured event at entry and exit with node name, action, latency, and document count.

#### `app/observability/tracer.py`

Maintains `state["trace"]` — a list of node names visited during a single request. Exposed in the `/ask` response and SSE `done` event.

#### `app/observability/debug.py`

Helpers for formatting the debug payload returned in the Web UI debug panel.

---

## Environment variable reference

Full list of variables recognised by `app/config.py` (all are optional unless marked required):

```bash
# LLM
GROQ_API_KEY=<your-key>          # REQUIRED for real LLM calls
GROQ_MODEL=openai/gpt-oss-120b  # any Groq model with structured-output support
USE_REAL_LLM=true                # false = mock mode (tests only)

# Retrieval
RETRIEVAL_MODE=lightweight_hybrid   # vector_only | lightweight_hybrid | true_hybrid
RETRIEVAL_USE_RERANKER=false        # true to enable cross-encoder (experimental)
RETRIEVAL_TOP_K=6
RETRIEVAL_HYBRID_VECTOR_K=12
RETRIEVAL_RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

# Embeddings
EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Data paths (relative to repo root)
CHROMA_PERSIST_DIR=data/chroma
CHROMA_COLLECTION=papers

# Docker bootstrap
BOOTSTRAP_INGEST=true
INGEST_QUERY=cat:cs.AI
INGEST_MAX_RESULTS=20
RESET_CHROMA=false
```
