"""Needle 2 runtime client — tool determination only (does not execute tools)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Protocol


class NeedleEngine(Protocol):
    def complete(self, text: str, max_new_tokens: int = 256) -> dict[str, Any]: ...

    def reset(self) -> None: ...


@dataclass
class NeedleCompleteResult:
    function_calls: list[dict[str, Any]]
    confidence: float | None
    reasoning: str | None
    raw: dict[str, Any]


class NeedleToolRuntime:
    """Wraps cactus-needle ``Needle.complete`` for schema-constrained tool calls."""

    def __init__(
        self,
        *,
        lib_path: str | None = None,
        factory: Any | None = None,
    ) -> None:
        self.lib_path = (
            lib_path
            or os.environ.get("EVPRTR_NEEDLE_LIB_PATH")
            or os.environ.get("NEEDLE_LIB_PATH")
        )
        self._factory = factory
        self._agent: Any | None = None
        self._bound_tools_key: str | None = None

    def available(self) -> bool:
        if self._factory is not None:
            return True
        if not self.lib_path:
            return False
        try:
            import needle  # noqa: F401
        except ImportError:
            return False
        return os.path.isfile(self.lib_path)

    def complete(
        self,
        query: str,
        tools: list[dict[str, Any]],
        *,
        max_new_tokens: int | None = None,
    ) -> NeedleCompleteResult:
        agent = self._agent_for(tools)
        agent.reset()
        tokens = max_new_tokens
        if tokens is None:
            tokens = int(os.environ.get("EVPRTR_NEEDLE_MAX_NEW_TOKENS", "1024") or "1024")
        raw = agent.complete(query, max_new_tokens=max(64, tokens))
        calls = raw.get("function_calls") or []
        if not isinstance(calls, list):
            calls = []
        conf = raw.get("confidence")
        return NeedleCompleteResult(
            function_calls=[c for c in calls if isinstance(c, dict)],
            confidence=float(conf) if conf is not None else None,
            reasoning=raw.get("reasoning"),
            raw=raw if isinstance(raw, dict) else {},
        )

    def _agent_for(self, tools: list[dict[str, Any]]) -> Any:
        key = repr(tools)
        if self._agent is not None and self._bound_tools_key == key:
            return self._agent
        if self.lib_path:
            os.environ["NEEDLE_LIB_PATH"] = self.lib_path
        if self._factory is not None:
            agent = self._factory(tools=tools)
        else:
            import needle

            agent = needle.Needle(tools=tools)
        self._agent = agent
        self._bound_tools_key = key
        return agent
