from __future__ import annotations

import json

from app.agent.state import AgentState
from app.observability.logger import get_logger
from app.tools.registry import TOOL_REGISTRY


_logger = get_logger("agent.tool")


def _normalize_tool_args(name: str, raw_args: dict) -> dict:
    args = {k: v for k, v in raw_args.items() if v is not None}

    if name == "arxiv_search":
        if "query" not in args and "expression" in args:
            args["query"] = args["expression"]
        return {k: v for k, v in args.items() if k in {"query", "max_results"}}

    if name == "calculator":
        if "expression" not in args and "query" in args:
            args["expression"] = args["query"]
        return {k: v for k, v in args.items() if k == "expression"}

    return args


def tool(state: AgentState) -> dict:
    state.setdefault("trace", []).append("tool")

    decision = state.get("decision") or {}
    name = decision.get("tool_name")
    args = _normalize_tool_args(name or "", decision.get("tool_args") or {})

    if not name or name not in TOOL_REGISTRY:
        _logger.warning("tool.unknown_tool name=%s", name)
        return {
            "documents": [],
            "answer": (
                f"I tried to use a tool named '{name}' but it isn't registered. "
                "Available tools: " + ", ".join(sorted(TOOL_REGISTRY.keys())) + "."
            ),
        }

    try:
        result = TOOL_REGISTRY[name](**args)
        _logger.info("tool.ok name=%s args_keys=%s", name, sorted(args.keys()))
    except TypeError as exc:
        _logger.error("tool.bad_args name=%s error=%s", name, exc)
        return {
            "documents": [],
            "answer": f"Tool '{name}' rejected the arguments ({exc}).",
        }
    except Exception as exc:
        _logger.error("tool.failure name=%s error=%s", name, exc)
        return {
            "documents": [],
            "answer": f"Tool '{name}' failed: {exc}",
        }

    result_text = (
        result
        if isinstance(result, str)
        else json.dumps(result, ensure_ascii=False, indent=2)
    )

    docs = [
        {
            "id": f"tool:{name}",
            "text": result_text,
            "source": f"tool:{name}",
            "score": 1.0,
        }
    ]
    return {"documents": docs}
