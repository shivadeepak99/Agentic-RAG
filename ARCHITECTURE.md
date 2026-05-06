# Agentic RAG System — Architecture Diagrams

---

## 1. Full System Overview

```
                         +---------------------+
                         |    Groq LLM API     |
                         |  openai/gpt-oss-120b|
                         |  (routing + answers)|
                         +----------+----------+
                                    ^
                                    |
   +------------+    +--------------+--------------+   
   | Browser UI +--->|                             +
   |  (Web UI)  |    |       FastAPI Server        |    
   +------------+    |                             |   
                     |  POST  /ask                 |
   +------------+    |  POST  /ask/stream (SSE)    |
   |    CLI     +--->|  GET   /memory/{session_id} |
   | (run.py)   |    |  GET   /sessions            |
   +------------+    |  GET   /health              |
                     +--------------+--------------+
                                    |
                     +--------------v--------------+
                     |       Memory Store          |
                     |      (per session)          |
                     |-----------------------------|
                     | Conversation  last 12 turns |
                     | Episodic      LLM-compressed|
                     | Semantic      topic profile |
                     +--------------+--------------+
                                    | context injected
                                    v
                     +--------------+--------------+
                     |      Agent  (LangGraph)     |
                     |   see Diagram 2 for routing |
                     +--------------+--------------+
                           |               |
               +-----------+               +-----------+
               |                                       |
               v                                       v
  +------------+------------+          +--------------+-------------+
  |     Retrieval Engine    |          |            Tools           |
  |-------------------------|          |----------------------------|
  |  Vector Search (MiniLM) |          |  calculator  (AST-safe)   |
  |  BM25 Reranking         |          |  arxiv_search (live API)  |
  |  True Hybrid (RRF)      |          +----------------------------+
  |  Cross-Encoder (opt.)   |
  +------------+------------+
               |
               v
  +------------+------------+
  |     Chroma Vector DB    |
  |     data/chroma/        |
  |  (local, persistent)    |
  +-------------------------+
```

---

## 2. Agent Routing (LangGraph)

```
                          (START)
                             |
                             v
                    +--------+--------+
                    |     decide      |  <-- LLM call (strict JSON schema)
                    |   (router)      |      fallback: heuristic rules
                    +--------+--------+
                             |
          +--------+---------+---------+--------+
          |        |         |         |        |
          v        v         v         v        v
      +-------+ +------+ +------+ +-------+ +------+
      |retrieve| | tool | |clarify| |refuse| |answer|
      +---+---+ +--+---+ +--+---+ +---+---+ +--+---+
          |        |       |           |        |
          |        |       v           v        v
          |        |     (END)       (END)   +------+
          |        |                         | chat |
          v        v                         +--+---+
      +---+---------+---+                       |
      |       answer    |                       v
      |  (grounded resp)|                     (END)
      +-----------------+
               |
               v
             (END)

  Node responsibilities
  ---------------------
  decide    calls LLM with memory context, returns one of 5 actions
  retrieve  runs hybrid_search(query) -> top-k chunks from Chroma
  tool      executes calculator or arxiv_search, returns result
  answer    synthesises grounded response from chunks or tool result
  chat      handles greetings and meta questions (no retrieval)
  clarify   asks user to be more specific
  refuse    declines sensitive or credential requests
```

---

## 3. Ingestion Pipeline

```
  $ python scripts/run_ingestion.py --query "cat:cs.AI" --max-results 150

          +------------------+
          |   arXiv Atom API |
          |  (metadata feed) |
          +--------+---------+
                   |  titles, abstracts, pdf_urls
                   v
          +--------+---------+
          |  fetch_arxiv()   |
          +--------+---------+
                   |
                   v
          +--------+---------+
          | download_pdf()   |  --> data/raw_pdfs/{id}.pdf
          +--------+---------+
                   |
                   v
          +--------+---------+
          | parse_pdfs()     |  --> data/parsed/{id}.txt
          | (PyMuPDF)        |
          +--------+---------+
                   |
                   v
          +--------+---------+
          | build_chunks()   |  --> data/chunks/{id}.chunks.json
          | 900 char window  |
          | 150 char overlap |
          +--------+---------+
                   |
                   v
          +--------+---------+
          | embed_texts()    |
          | MiniLM-L6-v2     |
          | (local, no API)  |
          +--------+---------+
                   |
                   v
          +--------+---------+
          |  Chroma DB       |  --> data/chroma/
          |  add_texts()     |
          +------------------+

  Also writes: data/metadata.json (paper titles, authors, arxiv ids)
  Docker path: BOOTSTRAP_INGEST=true triggers this automatically on startup
```

---

## 4. Memory System

```
                   +-----------------------------+
                   |      MemoryStore            |
                   |     (per session key)       |
                   +----+----------+----------+--+
                        |          |          |
                        v          v          v
             +----------+--+ +-----+-----+ +-+------------+
             | Conversation| | Episodic  | | Semantic     |
             |             | |           | |              |
             | deque of    | | LLM call  | | heuristic    |
             | last 12     | | compresses| | regex over   |
             | (user, asst)| | turns >   | | each turn:   |
             | turn pairs  | | 1200 chars| | topics,      |
             |             | | into ~800 | | preferences, |
             |             | | char digest| | entities    |
             +----------+--+ +-----+-----+ +-+------------+
                        |          |          |
                        +----------+----------+
                                   |
                        injected into every LLM call
                                   |
                     +-------------+-------------+
                     |                           |
                     v                           v
              decide prompt               answer prompt
           [semantic memory]           [semantic memory]
           [episodic summary]          [episodic summary]
           [recent turns]              [recent turns]
                                       [retrieved chunks]
```
