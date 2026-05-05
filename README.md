# Agentic RAG System

An agentic Retrieval-Augmented Generation system over arXiv cs.AI papers. The agent decides for itself whether to **retrieve**, **call a tool**, **ask for clarification**, **refuse**, or **answer directly**.

## Architecture

```
User → run_agent() → LangGraph StateGraph
                       │
                    [decide]  ← LLM router (Groq openai/gpt-oss-120b) + heuristic fallback
                    │  │  │  │  │
                    ▼  ▼  ▼  ▼  ▼
               retrieve clarify tool refuse answer
                    │              │
                    ▼              ▼
                 [answer]        [chat]
                    │              │
                  (END)          (END)
```

- **Agent brain**: 7-node LangGraph (`decide`, `retrieve`, `tool`, `clarify`, `refuse`, `answer`, `chat`).
- **Chat mode**: `answer` actions route to a dedicated `chat` node so greetings/meta questions don't get corpus-related disclaimers.
- **Retrieval**: Hybrid (semantic vector via `sentence-transformers/all-MiniLM-L6-v2` + BM25 reranking) over a persistent Chroma collection.
- **Memory**: Session-keyed sliding-window conversation memory **plus** an LLM-summarized rolling memory of older turns. Both are injected into the `decide`, `answer`, and `chat` prompts.
- **Tools**: Safe AST-based `calculator`, live `arxiv_search` against the public arXiv API.
- **Evaluation**: 15 hand-written cases covering retrieval, tool routing, clarification, refusal, OOD, and smalltalk — scored on action-correctness, behavior markers, and content keywords.
- **Observability**: Structured JSON logs at every node + a per-request `trace`/`decision`/`documents` payload returned by the `/ask` API.

## Memory types (per the assignment rubric)

- **Conversation memory (short-term / working memory)**: the last N turns verbatim (recency + exact phrasing). Implemented by [app/memory/conversation.py](app/memory/conversation.py) and stored per-session in [app/memory/store.py](app/memory/store.py).
- **Episodic memory (longer-horizon, compressed)**: a rolling summary of older conversation turns that survives beyond the sliding window. Implemented by [app/memory/summary.py](app/memory/summary.py).
- **Semantic memory**: durable facts/preferences extracted from the user and stored for later recall (often as a key-value store or a vector index of “user facts”). This system does **not** persist a separate semantic-user-memory store by default; the *paper corpus* itself functions as semantic memory for the domain knowledge.

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate                # Windows
# source .venv/bin/activate            # macOS/Linux

pip install -r requirements.txt
cp .env.example .env                   # then set GROQ_API_KEY
```

### Ingest a corpus

```bash
python scripts/run_ingestion.py --query "cat:cs.AI" --max-results 50
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

### Run evaluation

```bash
python scripts/run_eval.py
```

Writes per-case results to `data/eval/results.json` and prints aggregate metrics: average composite score and **action accuracy** (fraction of cases where the router picked the expected action).

### Hybrid retrieval ablation

```bash
python scripts/run_ablation.py
```

Runs the eval dataset twice — once with hybrid (vector + BM25 rerank) and once with vector-only — then prints a side-by-side score table showing the lift from BM25 reranking.

### Tests

```bash
pytest -q
```

## Decisions log

| Decision | What I considered | What I picked & why |
|---|---|---|
| Agent framework | LangGraph, LlamaIndex, raw orchestration | **LangGraph** — explicit `StateGraph`, conditional edges fit a 5-way router cleanly, easy to debug with the LangGraph Studio integration declared in `langgraph.json`. |
| LLM provider | OpenAI, Anthropic, Groq, local | **Groq `openai/gpt-oss-120b`** — fast inference, large context, and strict JSON-schema structured outputs for routing decisions. |
| Embeddings | OpenAI text-embedding-3, BGE, MiniLM, hash | **`sentence-transformers/all-MiniLM-L6-v2`** — runs locally for free, 384 dims, ~22MB, well-benchmarked. The repo also contains a SHA256 hash fallback so it degrades gracefully if the model can't be loaded (used only as a no-network safety net). |
| Vector store | FAISS, Chroma, Qdrant, in-memory | **Chroma `PersistentClient`** — local persistence, simple API, suffix collection name with `_semantic`/`_hash` so a model swap doesn't poison an existing index. |
| Retrieval technique | Top-k cosine only, hybrid, reranking, HyDE, multi-query | **Hybrid (vector + BM25 rerank)** — vector recalls semantically related chunks, BM25 then reranks the candidate pool to lift exact-keyword matches. Weights `0.7 * vec + 0.3 * bm25_norm`. Run `python scripts/run_ablation.py` to see the scored comparison against vector-only. |
| Chunking | Sentence-aware, recursive char splitter, fixed window | **Fixed 900-char window with 150-char overlap** — predictable, language-agnostic, fast. Acknowledged limitation: occasionally splits sentences. |
| Memory | None, sliding-window only, summary-only, hybrid | **Hybrid**: a deque of the last 12 turns (recency) + an LLM-summarized rolling memory that compresses to ~1.5K chars when it grows past 3K (long-horizon recall). Both feed the `decide` and `answer` prompts. Falls back to tail-truncation when offline. |
| Routing decision | Pure LLM, pure rules, hybrid | **LLM-primary, heuristic fallback** — Groq returns a JSON action; if the call or parse fails, a deterministic `_heuristic_decision()` covers refusal triggers, vague phrases, calculator detection, and `arxiv` keywords. The system is therefore never bricked by a transient API issue. |
| Eval scoring | Substring match only, exact match, LLM-as-judge | **Composite per-case**: action-correctness, behavior markers (refusal / clarification / "I don't know" phrasing for OOD), and content keyword presence. Aggregate reports both `avg_score` and `action_accuracy`. LLM-as-judge intentionally skipped to keep eval deterministic and free. |
| Observability | None, ad-hoc print, structured JSON, full tracing | **Structured JSON logs per node** + a `trace` field exposed via the API. LangSmith hooks not added by default but trivially enabled by env vars. |

## Failure modes observed

- **Empty corpus / no relevant chunk** → `hybrid_search` returns `[]`; for in-domain questions the assistant answers from general technical knowledge (without citations), and for clearly out-of-domain questions it responds with an honest "I don't know"-style message.
- **LLM API failure** in `decide` → falls back to `_heuristic_decision()`. Logged as `decide.fallback_heuristic`.
- **LLM API failure** in `answer` → returns a graceful message; if at least one document was retrieved, surfaces the top passage so the user still gets value.
- **Tool failure / unknown tool / bad args** → `tool` node returns a human-readable error rather than crashing the graph.
- **Vague follow-ups** (e.g. "expand on that") → routed to `clarify` either by the LLM or by the heuristic vague-phrase list.
- **Chroma unavailable** → `VectorStore` falls back to in-memory list with manual cosine similarity. Persistence is lost but the system still responds.

## What I'd do with another week

1. **Add a cross-encoder reranker** (`bge-reranker-v2-m3`) as a third retrieval stage after BM25 reranking and measure the lift via `scripts/run_ablation.py`.
2. **Query rewriting + multi-query retrieval**: prepend a small LLM step that produces 3 paraphrases per question, retrieve for each, then deduplicate by chunk ID. Particularly helpful for technical jargon mismatches.
3. **Parent-doc retrieval**: keep small chunks for matching but return their parent paragraph to the answer node — narrows recall without sacrificing context.
4. **LLM-as-judge eval track**: add a second eval pass that uses Groq to score faithfulness vs. retrieved context, complementing the deterministic substring/action checks.
5. **Per-paper metadata filtering**: when the user mentions a specific paper, filter the Chroma query by `source` rather than relying on the embedding to surface it.
6. **LangSmith / OTLP traces** wired in so each node's prompt and output is inspectable end-to-end.

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

## LangGraph Studio

```bash
langgraph dev --no-browser
```

Then open the Studio link it prints. The graph is registered as `agent` in `langgraph.json`.
