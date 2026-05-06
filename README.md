# Agentic RAG System

An agentic Retrieval-Augmented Generation system over arXiv cs.AI papers. The agent decides for itself whether to **retrieve**, **call a tool**, **ask for clarification**, **refuse**, or **answer directly**.

## Agent Architecture

<img width="1298" height="940" alt="image" src="https://github.com/user-attachments/assets/f401d382-1404-45ff-afa4-6c72b8de0e60" />


- **Agent brain**: 7-node LangGraph (`decide`, `retrieve`, `tool`, `clarify`, `refuse`, `answer`, `chat`).
- **Chat mode**: When `decide` returns `action: "answer"` (greetings, meta questions), the graph routes to the `chat` node — not the `answer` node. The `answer` node is reserved for synthesising retrieved corpus chunks or tool results. This keeps greetings and capability questions free of corpus-related disclaimers and citation rules. The naming is intentional: `action: "answer"` means "respond directly without retrieval"; `node: answer` means "generate a grounded response from context". Both are distinct and correct.
- **Retrieval**: Default is **lightweight hybrid**: semantic vector retrieval via `sentence-transformers/all-MiniLM-L6-v2` plus BM25 reranking over the vector candidate pool. A **true hybrid** fusion path and optional **cross-encoder reranker** are implemented behind config toggles and kept off by default because the current ablation did not show a win.
- **Memory**: Session-keyed sliding-window conversation memory **plus** an LLM-summarized rolling memory of older turns. Both are injected into the `decide`, `answer`, and `chat` prompts.
- **Tools**: Safe AST-based `calculator`, live `arxiv_search` against the public arXiv API.
- **Evaluation**: 17 hand-written cases covering retrieval, tool routing, clarification, refusal, OOD, and smalltalk — scored on action-correctness, behavior markers, and content keywords.
- **Observability**: Structured JSON logs at every node + a per-request `trace`/`decision`/`documents` payload returned by the `/ask` API.

## Memory types (per the assignment rubric)

All three types are implemented and injected into every `decide` and `answer` LLM call, labeled separately so the model can weight them appropriately.

- **Conversation memory (short-term / working memory)**: the last 12 turns verbatim — exact phrasing, order preserved. Implemented by [`app/memory/conversation.py`](app/memory/conversation.py). Gives the agent the raw transcript of the recent session.

- **Episodic memory (longer-horizon, compressed)**: a rolling LLM-summarized digest of older turns that would otherwise fall off the conversation window. Implemented by [`app/memory/summary.py`](app/memory/summary.py). Compresses to ~800 chars when the session transcript exceeds ~1200 chars. Falls back to tail truncation when the API is unavailable.

- **Semantic memory (structured user-profile facts)**: per-session structured knowledge about *this user* — which AI topics they've asked about, inferred preferences (e.g. “prefers code examples”), and recently mentioned entities. Implemented by [`app/memory/semantic.py`](app/memory/semantic.py). Extracted heuristically from each turn with no extra LLM call cost. Exposed via the `/memory/{session_id}` API and the in-app Memory panel. Unlike conversation and episodic memory (which are raw text), semantic memory stores *typed facts* that persist across topic switches within a session.

  Note: the arXiv corpus is the system's *domain* semantic memory — what it knows about the world. The `SemanticMemory` class is the *user-profile* semantic memory — what it knows about the person it's talking to. These are intentionally distinct.

## Quickstart

```text
+----------------------------------------------------------------------------------+
|                               RUNNING THE SYSTEM                                 |
+----------------------------------------------------------------------------------+
|  Option 1: Manual commands                                                       |
|    Best when your machine already has heavyweight ML dependencies installed.      |
|    Faster for iteration because you reuse your local Python environment.          |
|                                                                                  |
|  Option 2: Docker Compose                                                        |
|    Best for one-command startup and reproducibility.                             |
|    Slower on first run because Docker installs everything from scratch           |
|    inside the container, including large ML dependencies.                        |
+----------------------------------------------------------------------------------+
|  Important note                                                                  |
|    This project uses large dependencies such as PyTorch and sentence-transformers |
|    for local embeddings and optional reranking. First-time setup can take a      |
|    while, especially in Docker. If you already have these installed locally,     |
|    prefer the manual path for a faster startup.                                  |
+----------------------------------------------------------------------------------+
```

### Option 1 — Manual commands (fastest if dependencies are already installed)

```bash
pip install -r requirements.txt
cp .env.example .env                   # then set GROQ_API_KEY
```

Why this path is often faster:

- if PyTorch, `sentence-transformers`, and related ML dependencies are already installed on your machine, you avoid reinstalling them inside a fresh container
- local iteration is usually quicker for repeated ingestion / eval / ablation runs
- it is the easiest path for development and debugging

### Ingest a corpus

```bash
python scripts/run_ingestion.py --query "cat:cs.AI" --max-results 150
```

This fetches arXiv metadata, downloads PDFs, parses with PyMuPDF, chunks (900-char window with 150-char overlap), embeds with MiniLM, and stores in Chroma at `data/chroma/`.

### Ask questions

CLI (multi-turn, with memory):

```bash
python run.py --chat --session demo --show-trace
```

CLI (one-shot):

```bash
python run.py "Explain attention in transformers"
```

API:

```bash
uvicorn app.main:app --reload
# POST http://127.0.0.1:8000/ask  { "question": "...", "session_id": "demo" }
```

The `/ask` response includes `answer`, `trace`, `decision`, `documents`, and `session_id`.

### Option 2 — One-click Docker run

Copy the env template, set your flags, then start everything with one command:

```bash
cp .env.example .env
# set GROQ_API_KEY if you want real LLM calls

docker compose up --build
```

What this does:

- builds the app image
- optionally bootstraps ingestion on first startup
- persists Chroma / parsed data under `./data`
- starts the FastAPI app on `http://127.0.0.1:8000`

Why this path is slower on the first run:

- Docker builds a fresh environment inside the container
- that means reinstalling large packages like PyTorch and `sentence-transformers`
- model downloads and ingestion bootstrap can also add noticeable startup time

Use Docker when you want the cleanest reproducible setup. Use the manual path when you already have the heavy dependencies installed and want the fastest startup.

Useful container flags in `.env`:

- `BOOTSTRAP_INGEST=true|false` — run ingestion automatically before the server starts
- `INGEST_QUERY=cat:cs.AI` — arXiv query used during bootstrap
- `INGEST_MAX_RESULTS=20` — number of papers to fetch on bootstrap
- `RESET_CHROMA=true|false` — rebuild the vector index on container start
- `GROQ_MODEL=...` — choose the LLM model
- `USE_REAL_LLM=true|false` — force live Groq calls or allow mock mode
- `RETRIEVAL_MODE=lightweight_hybrid|vector_only|true_hybrid`
- `RETRIEVAL_USE_RERANKER=true|false`

For a fast demo startup, the defaults are intentionally conservative. If you want a larger corpus, raise `INGEST_MAX_RESULTS` and rerun:

```bash
docker compose up --build
```

### Run evaluation

```bash
python scripts/run_eval.py
```

Writes per-case results to `data/eval/results.json` and prints aggregate metrics: average composite score and **action accuracy** (fraction of cases where the router picked the expected action).

> **Eval requirements**: set `GROQ_API_KEY` and `USE_REAL_LLM=true` before running — mock mode produces canned responses that will not exercise real routing or content quality. The published results were generated with `GROQ_MODEL=openai/gpt-oss-120b` on a corpus of ~150 arXiv cs.AI papers ingested via `python scripts/run_ingestion.py --query "cat:cs.AI" --max-results 150`.

### Retrieval ablation

```bash
python scripts/run_ablation.py
```

Runs the eval dataset across four retrieval modes:

- `vector-only`
- `lightweight hybrid` (current default)
- `true hybrid (no reranker)`
- `true hybrid + cross-encoder`

The script prints both the full-agent score table and the retrieval-sensitive subset. Full results are in [`ablationresults.md`](ablationresults.md).

**Key findings** (see [`ablationresults.md`](ablationresults.md) for full per-case breakdown):

- **Lightweight hybrid (0.986) is the best overall mode**, edging vector-only (0.982) by +0.004 on the full 18-case eval. The BM25 reranking pass over vector candidates adds a consistent small gain.
- **Cross-encoder reranker**: helps within the true-hybrid family (+0.019 over true hybrid without reranker on the retrieval subset), but does not beat lightweight hybrid. Disabled by default because the compute cost of loading and scoring a CrossEncoder is not justified relative to the cheaper lightweight path.
- **All four modes achieve 100% action accuracy** — retrieval strategy does not affect routing decisions.

> **Earlier results in git history** were generated by a buggy version of the script that called the `hybrid_search` dispatcher for the "true hybrid + cross-encoder" column. With the default `RETRIEVAL_MODE=lightweight_hybrid`, that column was actually running lightweight hybrid twice. The numbers above are from the corrected script.

### Tests

```bash
pytest -q
```

## Decisions log

| Decision | What I considered | What I picked & why |
|---|---|---|
| Agent framework | LangGraph, LlamaIndex, raw orchestration | **LangGraph** — explicit `StateGraph`, conditional edges fit a 5-way router cleanly, easy to debug with the LangGraph Studio integration declared in `langgraph.json`. |
| LLM provider | OpenAI, Anthropic, Groq, local | **Groq `openai/gpt-oss-120b`** — fast inference (≈500 tok/s), 131k context, and strict JSON-schema constrained decoding for routing decisions. The model is OpenAI's open-weights release served through Groq's OpenAI-compatible endpoint; any Groq-hosted model with structured-output support (e.g. `llama-3.3-70b-versatile`) could substitute by swapping `GROQ_MODEL` in `.env`. Groq was chosen over Anthropic for this project because constrained JSON-schema decoding — which makes the routing decision deterministic — is supported on Groq today; Claude's tool-use API achieves a similar result but requires a different integration pattern. |
| Embeddings | OpenAI text-embedding-3, BGE, MiniLM, hash | **`sentence-transformers/all-MiniLM-L6-v2`** — runs locally for free, 384 dims, ~22MB, well-benchmarked. The repo also contains a SHA256 hash fallback so it degrades gracefully if the model can't be loaded (used only as a no-network safety net). |
| Vector store | FAISS, Chroma, Qdrant, in-memory | **Chroma `PersistentClient`** — local persistence, simple API, suffix collection name with `_semantic`/`_hash` so a model swap doesn't poison an existing index. |
| Retrieval technique | Top-k cosine only, lightweight hybrid, true hybrid fusion, cross-encoder reranking | **Default: lightweight hybrid (vector + BM25 rerank over vector hits)** — the ablation confirms this is the best overall mode (0.986 avg score vs 0.982 for vector-only). I also implemented **true hybrid** (independent vector + BM25 with RRF fusion) and an optional **cross-encoder reranker**. The reranker adds +0.019 within the true-hybrid family but does not beat lightweight hybrid, so it stays disabled by default; the compute cost of a CrossEncoder inference pass is not justified by the marginal gain. All modes available via `RETRIEVAL_MODE`. |
| Chunking | Sentence-aware, recursive char splitter, fixed window | **Fixed 900-char window with 150-char overlap** — predictable, language-agnostic, fast. Acknowledged limitation: occasionally splits sentences. |
| Memory | None, sliding-window only, summary-only, three-type hybrid | **Three-type hybrid**: (1) conversation memory — deque of last 12 verbatim turns; (2) episodic memory — LLM-compressed digest of older turns, triggered at ~1200 chars; (3) semantic memory — structured user-profile facts (topics, preferences, entities) extracted heuristically per turn. All three are labeled and injected into `decide` and `answer` prompts. The distinction matters: conversation gives recency, episodic gives long-horizon coherence, semantic gives user-level personalization without re-reading the full transcript. |
| Routing decision | Pure LLM, pure rules, hybrid | **LLM-primary, heuristic fallback** — Groq returns a JSON action; if the call or parse fails, a deterministic `_heuristic_decision()` covers refusal triggers, vague phrases, calculator detection, and `arxiv` keywords. The system is therefore never bricked by a transient API issue. |
| Eval scoring | Substring match only, exact match, LLM-as-judge | **Composite per-case**: action-correctness, behavior markers (refusal / clarification / "I don't know" phrasing for OOD), and content keyword presence. Aggregate reports both `avg_score` and `action_accuracy`. LLM-as-judge intentionally skipped to keep eval deterministic and free. |
| Observability | None, ad-hoc print, structured JSON, full tracing | **Structured JSON logs per node** + a `trace` field exposed via the API. LangSmith hooks not added by default but trivially enabled by env vars. |

## Failure modes observed

- **Empty corpus / no relevant chunk** → the answer prompt is grounded-first and responds with an honest "I don't know based on available documents" style answer rather than fabricating a citation-backed response.
- **Retrieved chunks contradict each other** → the `ANSWER_SYSTEM` prompt instructs the model to acknowledge the conflict explicitly and explain only what each cited chunk individually supports, rather than synthesising a false consensus. This is a prompt-level guarantee; a future eval would need crafted corpus passages with planted contradictions to verify it empirically.
- **LLM API failure** in `decide` → falls back to `_heuristic_decision()`. Logged as `decide.fallback_heuristic`.
- **LLM API failure** in `answer` → returns a graceful message; if at least one document was retrieved, surfaces the top passage so the user still gets value.
- **Tool failure / unknown tool / bad args** → `tool` node returns a human-readable error rather than crashing the graph.
- **Vague follow-ups** (e.g. "expand on that") → routed to `clarify` either by the LLM or by the heuristic vague-phrase list.
- **Chroma unavailable** → `VectorStore` falls back to in-memory list with manual cosine similarity. Persistence is lost but the system still responds.

## What I'd do with another week

1. **Retrieval-only evaluation**: add labeled relevance judgments (expected chunk ids / paper ids) so retrieval tuning is measured directly, not only through final-answer wording.
2. **Retrieval confidence gating**: add a hard low-confidence path so clearly irrelevant/OOD retrieval results do not get passed to the answer node as if they were useful evidence.
3. **Query rewriting + multi-query retrieval**: prepend a small LLM step that produces 3 paraphrases per question, retrieve for each, then deduplicate by chunk ID. Particularly helpful for technical jargon mismatches.
4. **Parent-doc retrieval**: keep small chunks for matching but return their parent paragraph to the answer node — narrows recall without sacrificing context.
5. **Per-paper metadata filtering**: when the user mentions a specific paper, filter retrieval by `source` rather than relying on the embedding to surface it.
6. **Domain-tuned reranker or calibrated reranker gating**: revisit reranking only if a stronger model or better evaluation proves it helps this corpus.

## Known limitations

- Chunking is a fixed sliding window; it can split mid-sentence on poorly-formatted PDFs.
- The mock LLM path (used when `GROQ_API_KEY` is empty) produces canned responses — useful for tests but not for evaluation. Evals should always be run with the real API key set.
- Memory is in-process and per-server-instance; restarting the API loses session state. A real deployment would back `MemoryStore` with Redis or SQLite.
- Calculator is integers/floats only — no symbolic math, no functions.
- arXiv tool uses lightweight string parsing of the Atom feed; sufficient in practice but a real `feedparser` would be more robust.

## Environment

Copy `.env.example` to `.env` and set:

- `GROQ_API_KEY` — required for real LLM calls.
- `GROQ_MODEL` — default `openai/gpt-oss-120b`.
- `EMBED_MODEL` — optional override of the sentence-transformers model.
- `RETRIEVAL_MODE` — one of `lightweight_hybrid` (default), `vector_only`, or `true_hybrid`.
- `RETRIEVAL_USE_RERANKER` — `false` by default; set `true` only for experiments with the cross-encoder path.
- `BOOTSTRAP_INGEST` — `true` by default in Docker flow; ingests papers before the server starts if no persisted index exists.
- `INGEST_QUERY` — bootstrap arXiv query, default `cat:cs.AI`.
- `INGEST_MAX_RESULTS` — bootstrap paper count, default `20`.
- `RESET_CHROMA` — set `true` to force a clean reindex on the next container start.

## LangGraph Studio

```bash
langgraph dev --no-browser
```

Then open the Studio link it prints. The graph is registered as `agent` in `langgraph.json`.
