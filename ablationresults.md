# Ablation Results

This document captures evaluation outputs across retrieval configurations.

## What the retrieval modes mean (in this codebase)

These labels correspond to the behavior in `app/retrieval/hybrid.py`:

### vector-only

Retrieval is purely semantic/vector search.

- The system embeds the query and searches the vector store (Chroma if available, otherwise an in-memory fallback).
- Results are scored using vector similarity (Chroma distance is converted to a score; fallback uses dot-product/cosine-like similarity).

### lightweight hybrid

“Hybrid” in a lightweight sense: it starts with vector search, then uses BM25 only to rerank those vector hits.

- Step 1: Get top vector candidates.
- Step 2: Build a small BM25 index over *only those candidate texts*.
- Step 3: BM25-score the candidates, normalize BM25 by the max BM25 score, then blend:
	- `final_score = 0.7 * vector_score + 0.3 * bm25_normalized`
- This keeps the candidate set constrained to what vector search surfaced, and uses keyword overlap to reshuffle within that set.

### true hybrid

“True” hybrid means it retrieves from two independent sources (vector + lexical BM25 corpus) and fuses the ranked lists.

- Step 1: Get vector candidates (semantic search).
- Step 2: Get BM25 candidates from a corpus built from chunk files in `data/chunks/*.chunks.json` (or from the vector store as a fallback).
- Step 3: Fuse both ranked lists using Reciprocal Rank Fusion (RRF): each item’s score is the sum of `1/(k + rank)` contributions from each list (here `k = 60`).
- Output is then truncated to the configured top-k.

### true hybrid + cross-encoder

This is true hybrid (vector + BM25 + RRF fusion) plus an optional reranking stage.

- After fusion, the system can rerank the fused candidates using a sentence-transformers `CrossEncoder` (default model: `cross-encoder/ms-marco-MiniLM-L-6-v2`).
- The cross-encoder scores *(query, document)* pairs directly, and the code applies a sigmoid to convert raw scores into a 0–1-ish score before sorting.
- If the cross-encoder model cannot be loaded, reranking is skipped and the fused ranking is used.

## Full agent eval (includes tool/chat/refusal paths)

### vector-only
- avg_score:       0.985
- action_accuracy: 1.000  (17 cases with expected_action)

### lightweight hybrid
- avg_score:       0.975
- action_accuracy: 1.000  (17 cases with expected_action)

### true hybrid (no reranker)
- avg_score:       0.971
- action_accuracy: 1.000  (17 cases with expected_action)

### true hybrid + cross-encoder
- avg_score:       0.985
- action_accuracy: 1.000  (17 cases with expected_action)

## Retrieval-sensitive subset: true hybrid + cross-encoder vs. vector-only

### true hybrid + cross-encoder
- avg_score:       0.917
- action_accuracy: 1.000  (8 cases with expected_action)

### vector-only
- avg_score:       0.948
- action_accuracy: 1.000  (8 cases with expected_action)

```text
ID                     hybrid   vec-only     delta
----------------------------------------------------
k1                      0.750      0.750  +  0.000
k2                      1.000      1.000  +  0.000
k3                      1.000      1.000  +  0.000
k4                      1.000      1.000  +  0.000
k5                      1.000      1.000  +  0.000
k6                      0.750      1.000   -0.250
o1                      1.000      1.000  +  0.000
o2                      0.833      0.833  +  0.000
----------------------------------------------------
TOTAL                   0.917      0.948   -0.031
```

Hybrid underperforms vector-only by 0.031 avg score.

## Retrieval-sensitive subset: true hybrid + cross-encoder vs. lightweight hybrid

### true hybrid + cross-encoder
- avg_score:       0.917
- action_accuracy: 1.000  (8 cases with expected_action)

### lightweight hybrid
- avg_score:       0.948
- action_accuracy: 1.000  (8 cases with expected_action)

```text
ID                  true-hybrid+xenc   lw-hybrid     delta
------------------------------------------------------------
k1                             0.750       0.750  +  0.000
k2                             1.000       1.000  +  0.000
k3                             1.000       1.000  +  0.000
k4                             1.000       1.000  +  0.000
k5                             1.000       1.000  +  0.000
k6                             0.750       1.000   -0.250
o1                             1.000       1.000  +  0.000
o2                             0.833       0.833  +  0.000
------------------------------------------------------------
TOTAL                          0.917       0.948   -0.031
```

True hybrid + cross-encoder underperforms lightweight hybrid by 0.031 avg score.

## Retrieval-sensitive subset: true hybrid + cross-encoder vs. true hybrid without reranker

### true hybrid + cross-encoder
- avg_score:       0.917
- action_accuracy: 1.000  (8 cases with expected_action)

### true hybrid (no reranker)
- avg_score:       0.969
- action_accuracy: 1.000  (8 cases with expected_action)

```text
ID                  true-hybrid+xenc   true-hybrid     delta
-------------------------------------------------------------
k1                             0.750         0.750  +  0.000
k2                             1.000         1.000  +  0.000
k3                             1.000         1.000  +  0.000
k4                             1.000         1.000  +  0.000
k5                             1.000         1.000  +  0.000
k6                             0.750         1.000   -0.250
o1                             1.000         1.000  +  0.000
o2                             0.833         1.000   -0.167
-------------------------------------------------------------
TOTAL                          0.917         0.969   -0.052
```

True hybrid + cross-encoder underperforms true hybrid (no reranker) by 0.052 avg score.
The cross-encoder reranker hurts on this benchmark — it is disabled by default.
