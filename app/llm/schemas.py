from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


Action = Literal["retrieve", "clarify", "tool", "refuse", "answer"]


class Decision(BaseModel):
    action: Action = Field(description="Which node to run next")

    query: str | None = Field(default=None, description="Query for retrieval")
    tool_name: str | None = Field(default=None, description="Tool to execute")
    tool_args: dict[str, Any] = Field(default_factory=dict)

    def to_agent_decision(self) -> dict[str, Any]:
        data = {"action": self.action}
        if self.query:
            data["query"] = self.query
        if self.tool_name:
            data["tool_name"] = self.tool_name
            data["tool_args"] = self.tool_args
        return data
