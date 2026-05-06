"""Semantic memory — structured per-session facts extracted from conversation.

Semantic memory is the third memory type alongside:
  - Conversation memory  (short-term verbatim turns, ConversationMemory)
  - Episodic memory      (compressed older turns, SummaryMemory)
  - Semantic memory      (THIS FILE: durable user-level facts and topic interests)

Unlike conversation and episodic memory (which are raw text), semantic memory
stores *structured* knowledge about the user extracted from the session:
  - topics they've shown interest in
  - entities they've mentioned (paper names, authors, methods)
  - preferences ("prefers code examples", "working on legal RAG")

Implementation strategy:
  After each turn we extract entities/topics heuristically (no extra LLM call)
  from the question text.  When the session is long enough we optionally run
  a lightweight LLM summarisation to extract richer user-profile facts.

This is intentionally cheap — semantic memory updates must not slow down
the main request path.
"""
from __future__ import annotations

import re
from collections import Counter


# ---------------------------------------------------------------------------
# Heuristic entity / topic extraction (no LLM cost)
# ---------------------------------------------------------------------------

_TOPIC_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(transformer|attention|bert|gpt|llm|large language model)\b", re.I), "transformers"),
    (re.compile(r"\b(rag|retrieval.augmented|retrieval augmented)\b", re.I), "RAG"),
    (re.compile(r"\b(diffusion|stable.diffusion|ddpm)\b", re.I), "diffusion models"),
    (re.compile(r"\b(reinforcement.learning|rlhf|reward.model)\b", re.I), "reinforcement learning"),
    (re.compile(r"\b(embedding|vector.store|chroma|faiss|semantic.search)\b", re.I), "embeddings/vector search"),
    (re.compile(r"\b(chain.of.thought|cot|few.shot|prompt.engineer)\b", re.I), "prompting techniques"),
    (re.compile(r"\b(mixture.of.experts|moe|sparse.model)\b", re.I), "mixture of experts"),
    (re.compile(r"\b(computer.vision|image.generation|vision.language|vit)\b", re.I), "computer vision"),
    (re.compile(r"\b(graph.neural|gnn|knowledge.graph)\b", re.I), "graph neural networks"),
    (re.compile(r"\b(agent|agentic|tool.use|function.call)\b", re.I), "agentic systems"),
    (re.compile(r"\b(fine.tun|lora|qlora|peft|adapter)\b", re.I), "fine-tuning"),
    (re.compile(r"\b(code|python|programming|software)\b", re.I), "code/programming"),
]

_PREF_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bcode\b|\bexample\b|\bshow me\b|\bdemonstrat", re.I), "prefers code examples"),
    (re.compile(r"\bintuit|\bsimple|\beli5|\bexplain.*plain\b", re.I), "prefers simple explanations"),
    (re.compile(r"\bpaper|\bpublication|\bresearch|\bstudy\b", re.I), "interested in research papers"),
]


class SemanticMemory:
    """Structured user-level facts accumulated across a session.

    This models what a human assistant would remember *about* a user
    beyond the raw conversation transcript — their interests, preferences,
    and domain context.
    """

    def __init__(self) -> None:
        self._topic_counts: Counter[str] = Counter()
        self._preferences: set[str] = set()
        self._mentioned_entities: list[str] = []  # paper titles, authors, etc.
        self._turn_count: int = 0

    def update(self, user_text: str, assistant_text: str = "") -> None:
        """Extract facts from a new turn and add to the semantic store."""
        self._turn_count += 1
        combined = f"{user_text} {assistant_text}"

        for pattern, topic in _TOPIC_PATTERNS:
            if pattern.search(combined):
                self._topic_counts[topic] += 1

        for pattern, pref in _PREF_PATTERNS:
            if pattern.search(user_text):
                self._preferences.add(pref)

    def add_entity(self, entity: str) -> None:
        """Manually register a named entity (paper title, author, etc.)."""
        if entity and entity not in self._mentioned_entities:
            self._mentioned_entities.append(entity)

    def top_topics(self, n: int = 5) -> list[str]:
        return [t for t, _ in self._topic_counts.most_common(n)]

    def as_text(self) -> str:
        """Return a compact text block suitable for injection into prompts."""
        if self._turn_count == 0:
            return ""

        parts: list[str] = []

        topics = self.top_topics()
        if topics:
            parts.append(f"Topics user has asked about: {', '.join(topics)}.")

        if self._preferences:
            parts.append(f"User preferences inferred: {'; '.join(sorted(self._preferences))}.")

        if self._mentioned_entities:
            recent = self._mentioned_entities[-4:]
            parts.append(f"Entities mentioned recently: {', '.join(recent)}.")

        return "\n".join(parts) if parts else ""

    def as_dict(self) -> dict:
        return {
            "turn_count": self._turn_count,
            "top_topics": self.top_topics(),
            "preferences": sorted(self._preferences),
            "recent_entities": self._mentioned_entities[-4:],
        }
