# Agentic RAG System — Complete ASCII Architecture

This document is a *block-based* ASCII map of how the whole system works: runtime request flow, agent routing, retrieval + storage, ingestion/indexing, memory, observability, evaluation, and Docker startup.


## 1) Top-Level System (What talks to what)

```text
                                   +-----------------------------+
                                   |      External Services       |
                                   |-----------------------------|
                                   |  Groq LLM API (openai/gpt-oss-120b) |
                                   |  arXiv Atom API (metadata)   |
                                   |  arXiv PDF URLs (paper PDFs) |
                                   +--------------+--------------+
                                                  ^
                                                  |
+-------------------+   HTTP + SSE (optional)     |        +----------------------+
|      User         |-----------------------------+------->|  FastAPI app.main     |
|-------------------|                                     |----------------------|
| Browser UI (/)    |<------------------------------------|  GET  /              |
| REST client       |   JSON response / SSE tokens         |  GET  /health        |
| CLI (python run.py)|------------------------------------>|  GET  /sessions      |
+---------+---------+                                     |  GET  /memory/{sid}  |
          |                                               |  POST /ask           |
          | (direct function call)                        |  POST /ask/stream    |
          v                                               +----------+-----------+
+-------------------+                                                |
| CLI entrypoint    |                                                |
| run.py -> cli_main|                                                |
+---------+---------+                                                v
          |                                            +--------------------------+
          |                                            | MemoryStore (per session)|
          |                                            |--------------------------|
          |                                            | ConversationMemory (N=12)|
          |                                            | SummaryMemory (LLM compress)|
          |                                            | SemanticMemory (heuristics)|
          |                                            +------------+-------------+
          |                                                         |
          | inject: history/summary/(semantic*)                      |
          |                                                         v
          |                                            +--------------------------+
          +------------------------------------------->| Agent Graph (LangGraph)  |
                                                       | or fallback runner       |
                                                       +------------+-------------+
                                                                    |
                                                                    v
                                                       +--------------------------+
                                                       |   Agent Nodes / Router   |
                                                       |--------------------------|
                                                       | decide / retrieve / tool |
                                                       | answer / chat / clarify  |
                                                       | refuse                   |
                                                       +------------+-------------+
                                                                    |
                                                                    v
                                                       +--------------------------+
                                                       | Response + trace + stats |
                                                       +--------------------------+

Notes:
  * semantic memory is injected in HTTP /ask and /ask/stream; the CLI currently injects
    history+summary but not semantic memory.
```


## 2) Agent Routing (LangGraph StateGraph)

```text
                 (START)
                    |
                    v
               +---------+
               | decide  |  -> decision.action in {retrieve|tool|clarify|refuse|answer}
               +----+----+
                    |
        +-----------+-----------+-----------+-----------+-----------+
        |                       |                       |           |
        v                       v                       v           v
   +---------+              +--------+              +---------+   +--------+
   | retrieve|              |  tool  |              | clarify |   | refuse |
   +----+----+              +---+----+              +----+----+   +---+----+
        |                       |                        |            |
        v                       v                        v            v
   +---------+              +--------+                (END)        (END)
   | answer  |              | answer | 
   +----+----+              +---+----+
        |                       |
        v                       v
      (END)                   (END)

Special case:
  decision.action = "answer" routes to chat() (no retrieval grounding) and then END.
```


## 3) Runtime: Blocking REST path (`POST /ask`)

```text
+-------------------+
| Client (UI/REST)  |
| POST /ask {q,sid} |
+---------+---------+
          |
          v
+---------------------------+
| FastAPI: ask(req)         |
| - get_session(sid)         |
| - build initial AgentState |
|   {question, history,      |
|    memory_summary,         |
|    semantic_memory, trace} |
+-------------+-------------+
              |
              v
+---------------------------+
| run_agent(graph, state)   |
|  (LangGraph invoke OR      |
|   sequential fallback)     |
+-------------+-------------+
              |
              v
+---------------------------+
| decide(state)              |
| - LLMClient.complete_json  |
|   (Groq structured output) |
| - fallback: heuristics     |
+-------------+-------------+
              |
              v
      +-------+-----------------------------------------------+
      | decision.action                                       |
      |------------------------------------------------------|
      | retrieve -> retrieve() -> answer()                    |
      | tool     -> tool()     -> answer()                    |
      | answer   -> chat()                                    |
      | clarify  -> clarify()                                 |
      | refuse   -> refuse()                                  |
      +-------+-----------------------------------------------+
              |
              v
+---------------------------+
| session.add_turn(q, ans)  |
| - conversation memory      |
| - episodic summary (LLM)   |
| - semantic memory          |
+-------------+-------------+
              |
              v
+---------------------------+
| JSON response              |
| {answer, trace, decision,  |
|  documents(if debug),      |
|  llm_stats, session_id}    |
+---------------------------+

Side channels:
  - JSON logs to stdout from each node (observability/logger.py)
  - trace[] list appended at each node: ["decide","retrieve","answer",...]
```


## 4) Runtime: Streaming path (`POST /ask/stream`)

```text
+--------------------+
| Client (UI)        |
| POST /ask/stream   |
+---------+----------+
          |
          v
+-----------------------------+
| FastAPI: ask_stream(req)    |
| - get_session(sid)           |
| - build AgentState           |
| - decide(state)              |
+--------------+--------------+
               |
               v
      +--------+--------------------------------------------+
      | action                                               |
      |-----------------------------------------------------|
      | retrieve -> retrieve(state) -> answer_stream(state)  |
      | tool     -> tool(state)     -> answer_stream(state)  |
      | answer   -> chat_stream(state)                       |
      | clarify  -> send one message, done                   |
      | refuse   -> send one message, done                   |
      +--------+--------------------------------------------+
               |
               v
+-----------------------------+
| SSE event stream             |
| - event: token  {text: ...}  |
| - event: debug  {decision/...} (optional)
| - event: done   {trace, llm_stats}
+-----------------------------+
               |
               v
+-----------------------------+
| session.add_turn(q, final)  |
+-----------------------------+
```


## 5) Retrieval Subsystem (Vector + BM25 + Optional Reranker)

```text
                    +---------------------+
Query -------------->| hybrid_search(q)   |
                    | (mode from settings)|
                    +----+-----------+----+
                         |           |
          mode=vector-only           | mode=lightweight_hybrid / true_hybrid
                         |           |
                         v           v
                +----------------+   +----------------------------------+
                | VectorStore    |   | Hybrid retrieval orchestration    |
                | search(q)      |   |----------------------------------|
                | (Chroma or mem)|   | (A) vector candidates            |
                +-------+--------+   | (B) BM25 candidates               |
                        |            | (C) fuse / rerank / trim          |
                        v            +----------------+-----------------+
              +-------------------+                   |
              | embed_text(q)     |                   |
              | SentenceTransformer|                  |
              | or hash fallback  |                  |
              +-------------------+                   |
                                                      v
                                         +-------------------------------+
                                         | (A) Vector candidates         |
                                         | store.search(q, top_k=Kvec)   |
                                         +---------------+---------------+
                                                         |
                                                         v
                                         +-------------------------------+
                                         | (B) BM25 candidates           |
                                         | LexicalCorpus.search(q,Kbm25) |
                                         +---------------+---------------+
                                                         |
                                                         v
                                         +-------------------------------+
                                         | (C) Fuse + optional reranker  |
                                         | - RRF fuse (vector + BM25)    |
                                         | - CrossEncoder rerank (opt)   |
                                         +-------------------------------+

LexicalCorpus data source:
  data/chunks/*.chunks.json  -> load text records -> BM25Index
  (fallback) if chunks missing -> VectorStore.all_docs()

VectorStore storage:
  - Primary: Chroma PersistentClient at data/chroma
  - Fallback: in-memory list if Chroma import/init fails
  - Collection name suffix depends on embedding mode:
      {collection}_semantic   (SentenceTransformer OK)
      {collection}_hash       (hash fallback)
```


## 6) Tools Subsystem (Explicit external operations)

```text
+--------------------+
| tool(state)        |
| tool_name + args   |
+---------+----------+
          |
          v
+-----------------------------+
| TOOL_REGISTRY               |
|-----------------------------|
| calculator(expression)      |
|   - safe AST eval           |
| arxiv_search(query,max=3)   |
|   - calls fetch_arxiv()     |
+---------+-------------------+
          |
          v
+-----------------------------+
| Tool result becomes a        |
| pseudo-document:             |
|  source="tool:<name>"        |
|  text=<result JSON/text>     |
+-----------------------------+
          |
          v
+-----------------------------+
| answer() formats it as       |
| [TOOL RESULT] ...            |
+-----------------------------+
```


## 7) Ingestion + Index Build (Corpus construction)

```text
Trigger paths:
  (A) Docker startup: scripts/container_boot.py
  (B) Manual:        python scripts/run_ingestion.py --query ... --max-results ...

(A) Container boot flow
----------------------
+-----------------------------+
| container_boot.py           |
| CMD in Dockerfile           |
+--------------+--------------+
               |
               v
+-----------------------------+
| Check persisted index        |
| data/chroma exists?          |
| BOOTSTRAP_INGEST?            |
| RESET_CHROMA?                |
+--------------+--------------+
               |
               +-------------------(skip)--------------------+
               |                                             |
               v                                             v
+-----------------------------+                    +---------------------------+
| Run ingestion pipeline      |                    | Start API server          |
| python scripts/run_ingestion|                    | uvicorn app.main:app      |
+--------------+--------------+                    +---------------------------+
               |
               v

(B) Ingestion pipeline detail
-----------------------------
+----------------------------+
| scripts/run_ingestion.py   |
+--------------+-------------+
               |
               v
+----------------------------+
| fetch_arxiv(query,max)     |----> arXiv Atom API
+--------------+-------------+
               |
               v
+----------------------------+
| For each paper:            |
|  download_pdf()            |----> PDF URL
|  pdf_to_text() (PyMuPDF)   |
|  chunk_text()              |
|  write *.chunks.json       |
+--------------+-------------+
               |
               v
+----------------------------+
| data artifacts written      |
|----------------------------|
| data/raw_pdfs/*.pdf         |
| data/parsed/*.txt           |
| data/chunks/*.chunks.json   |
| data/metadata.json          |
+--------------+-------------+
               |
               v
+----------------------------+
| ingest_default_chunks_dir()|
| -> VectorStore.add_texts() |
+--------------+-------------+
               |
               v
+----------------------------+
| embed_texts(chunks)        |
| SentenceTransformer OR hash|
+--------------+-------------+
               |
               v
+----------------------------+
| Chroma persistent index    |
| data/chroma/               |
+----------------------------+
```


## 8) Memory Subsystem (3 types)

```text
                 +---------------------------+
                 | MemoryStore (per session) |
                 +-------------+-------------+
                               |
                 +-------------+-------------+-------------------+
                 |                           |                   |
                 v                           v                   v
     +---------------------+     +----------------------+   +----------------------+
     | ConversationMemory  |     | SummaryMemory        |   | SemanticMemory       |
     | (last N turns)      |     | (rolling compression)|   | (structured facts)   |
     | deque(maxlen=12)    |     | may call LLM to      |   | heuristics: topics,  |
     +----------+----------+     | compress older turns |   | prefs, entities      |
                |                +----------+-----------+   +----------+-----------+
                |                           |                          |
                +-----------+---------------+--------------------------+
                            |
                            v
               Injected into prompts as "memory" blocks
               - decide(): semantic + summary + history
               - answer(): semantic + summary + history + retrieved context
               - chat(): summary + history (no retrieval context)
```


## 9) Observability (logs + debug payloads)

```text
+------------------------------+
| Node-level JSON logging      |
| app.observability.logger     |
| - prints {level,name,msg}    |
+--------------+---------------+
               |
               v
        stdout / container logs

+------------------------------+
| API debug surfaces           |
| - state.trace[]              |
| - decision JSON              |
| - llm_stats (last call)      |
| - documents (when debug=true)|
+------------------------------+
```


## 10) Evaluation + Ablation (offline quality checks)

```text
+------------------------------+
| scripts/run_eval.py          |
+--------------+---------------+
               |
               v
+------------------------------+
| DATASET (app.eval.dataset)   |
| list[{question, expected...}]|
+--------------+---------------+
               |
               v
+------------------------------+
| For each case: run_agent()   |
| score:                       |
| - action match               |
| - refusal/clarify markers    |
| - OOD "don't know" markers   |
| - keyword presence           |
+--------------+---------------+
               |
               v
+------------------------------+
| write results                |
| data/eval/results.json       |
+------------------------------+

Ablation:
  scripts/run_ablation.py monkey-patches hybrid_search() to compare modes:
    - vector-only
    - lightweight hybrid
    - true hybrid (no reranker)
    - true hybrid + cross-encoder
```


## 11) Docker Compose Deployment (single service)

```text
Host machine
-----------
+--------------------------------------------+
| docker compose up --build                  |
+--------------------+-----------------------+
                     |
                     v
+--------------------------------------------+
| Service: agentic-rag                        |
|--------------------------------------------|
| Image build: Dockerfile (python:3.12-slim) |
| Container CMD: python scripts/container_boot.py
| Ports:  host:8000 -> container:8000        |
| Volume: ./data  -> /app/data               |
| Env: GROQ_API_KEY, USE_REAL_LLM,           |
|      BOOTSTRAP_INGEST, INGEST_QUERY, ...   |
+--------------------+-----------------------+
                     |
                     v
+--------------------------------------------+
| Runtime inside container                    |
|--------------------------------------------|
| 1) (optional) ingestion bootstrap           |
| 2) uvicorn app.main:app (FastAPI)           |
| 3) reads/writes persisted artifacts in /app/data
+--------------------------------------------+
```
