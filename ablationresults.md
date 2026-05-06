# Ablation Results

This document captures evaluation outputs across retrieval configurations.
Results were generated with `GROQ_MODEL=openai/gpt-oss-120b`, `USE_REAL_LLM=true`,
on a corpus of ~150 arXiv cs.AI papers (18 eval cases including the `mem1` memory case).

## What the retrieval modes mean  

These labels correspond to the behavior in `app/retrieval/hybrid.py`:

### vector-only

Retrieval is purely semantic/vector search.

- The system embeds the query and searches the vector store (Chroma if available, otherwise an in-memory fallback).
- Results are scored using vector similarity (Chroma distance is converted to a score; fallback uses dot-product/cosine-like similarity).

### lightweight hybrid

"Hybrid" in a lightweight sense: it starts with vector search, then uses BM25 only to rerank those vector hits.

- Step 1: Get top vector candidates.
- Step 2: Build a small BM25 index over *only those candidate texts*.
- Step 3: BM25-score the candidates, normalize BM25 by the max BM25 score, then blend:
	- `final_score = 0.7 * vector_score + 0.3 * bm25_normalized`
- This keeps the candidate set constrained to what vector search surfaced, and uses keyword overlap to reshuffle within that set.

### true hybrid

"True" hybrid means it retrieves from two independent sources (vector + lexical BM25 corpus) and fuses the ranked lists.

- Step 1: Get vector candidates (semantic search).
- Step 2: Get BM25 candidates from a corpus built from chunk files in `data/chunks/*.chunks.json` (or from the vector store as a fallback).
- Step 3: Fuse both ranked lists using Reciprocal Rank Fusion (RRF): each item's score is the sum of `1/(k + rank)` contributions from each list (here `k = 60`).
- Output is then truncated to the configured top-k.

### true hybrid + cross-encoder

This is true hybrid (vector + BM25 + RRF fusion) plus an optional reranking stage.

- After fusion, the system reranks the fused candidates using a sentence-transformers `CrossEncoder` (default model: `cross-encoder/ms-marco-MiniLM-L-6-v2`).
- The cross-encoder scores *(query, document)* pairs directly, and the code applies a sigmoid to convert raw scores into a 0–1-ish score before sorting.
- If the cross-encoder model cannot be loaded, reranking is skipped and the fused ranking is used.

## Full agent eval (includes tool/chat/refusal paths)

18 cases total (17 original + `mem1` memory case).

### vector-only
- avg_score:       0.982
- action_accuracy: 1.000  (18 cases with expected_action)

### lightweight hybrid
- avg_score:       0.986
- action_accuracy: 1.000  (18 cases with expected_action)

### true hybrid (no reranker)
- avg_score:       0.975
- action_accuracy: 1.000  (18 cases with expected_action)

### true hybrid + cross-encoder
- avg_score:       0.977
- action_accuracy: 1.000  (18 cases with expected_action)

**Ranking: lw-hybrid (0.986) > th+xenc (0.977) > vec-only (0.982) > th-no-xenc (0.975)**

Lightweight hybrid is the best overall mode on this benchmark.

## Retrieval-sensitive subset (9 cases: k1–k6, o1, o2, mem1)

### true hybrid + cross-encoder vs. vector-only

#### true hybrid + cross-encoder
- avg_score:       0.963
- action_accuracy: 1.000  (9 cases with expected_action)

#### vector-only
- avg_score:       0.963
- action_accuracy: 1.000  (9 cases with expected_action)

```text
ID                   th+xenc  vec-only     delta
--------------------------------------------------
k1                     0.917     0.917  +  0.000
k2                     1.000     1.000  +  0.000
k3                     0.833     0.917   -0.084
k4                     1.000     1.000  +  0.000
k5                     0.917     0.833  +  0.084
k6                     1.000     1.000  +  0.000
o1                     1.000     1.000  +  0.000
o2                     1.000     1.000  +  0.000
mem1                   1.000     1.000  +  0.000
--------------------------------------------------
TOTAL                  0.963     0.963  +  0.000
```

True hybrid + cross-encoder ties vector-only. The cross-encoder wins k5, loses k3; net zero.

### true hybrid + cross-encoder vs. lightweight hybrid

#### true hybrid + cross-encoder
- avg_score:       0.963
- action_accuracy: 1.000  (9 cases with expected_action)

#### lightweight hybrid
- avg_score:       0.972
- action_accuracy: 1.000  (9 cases with expected_action)

```text
ID                   th+xenc  lw-hybrid     delta
---------------------------------------------------
k1                     0.917      0.917  +  0.000
k2                     1.000      1.000  +  0.000
k3                     0.833      0.917   -0.084
k4                     1.000      1.000  +  0.000
k5                     0.917      0.917  +  0.000
k6                     1.000      1.000  +  0.000
o1                     1.000      1.000  +  0.000
o2                     1.000      1.000  +  0.000
mem1                   1.000      1.000  +  0.000
---------------------------------------------------
TOTAL                  0.963      0.972   -0.009
```

True hybrid + cross-encoder underperforms lightweight hybrid by 0.009 avg score.
k3 is the only case where lightweight hybrid is better; all other cases tie.

### true hybrid + cross-encoder vs. true hybrid without reranker

#### true hybrid + cross-encoder
- avg_score:       0.963
- action_accuracy: 1.000  (9 cases with expected_action)

#### true hybrid (no reranker)
- avg_score:       0.944
- action_accuracy: 1.000  (9 cases with expected_action)

```text
ID                   th+xenc  th-no-xenc     delta
----------------------------------------------------
k1                     0.917       0.917  +  0.000
k2                     1.000       1.000  +  0.000
k3                     0.833       0.917   -0.084
k4                     1.000       1.000  +  0.000
k5                     0.917       0.833  +  0.084
k6                     1.000       0.833  +  0.167
o1                     1.000       1.000  +  0.000
o2                     1.000       1.000  +  0.000
mem1                   1.000       1.000  +  0.000
----------------------------------------------------
TOTAL                  0.963       0.944  +  0.019
```

True hybrid + cross-encoder outperforms true hybrid (no reranker) by 0.019 avg score.
The cross-encoder helps on k5 and k6 (attention/token-focus questions where reranking
surfaces more relevant passages) but hurts on k3 (diffusion models).

## Summary of findings

| Comparison | Winner | Delta |
|---|---|---|
| lw-hybrid vs vec-only (full eval) | lw-hybrid | +0.004 |
| th+xenc vs vec-only (retrieval subset) | tie | 0.000 |
| th+xenc vs lw-hybrid (retrieval subset) | lw-hybrid | +0.009 |
| th+xenc vs th-no-xenc (retrieval subset) | th+xenc | +0.019 |

**Conclusions:**
1. Lightweight hybrid is the best overall mode — BM25 reranking over vector candidates adds a small consistent gain (+0.004 on full eval) over pure vector search.
2. The cross-encoder reranker adds value when comparing within the true-hybrid family (+0.019 over fusion-only), but does not close the gap against lightweight hybrid on this benchmark.
3. The cross-encoder is disabled by default: the compute cost of loading and scoring with a CrossEncoder model is not justified by the marginal gain relative to the much cheaper lightweight hybrid path.
4. All four modes achieve 100% action accuracy — routing is robust regardless of retrieval strategy.
