"""Tool path: Needle determines calls; compositor presents OpenAI-compatible shape."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any

from compositor.runtimes.needle import NeedleCompleteResult, NeedleToolRuntime
from compositor.tools.convert import (
    filter_tools_for_choice,
    last_user_text,
    needle_calls_to_openai_message,
    openai_tools_to_needle,
    should_route_tools_to_needle,
)


@dataclass
class ToolPathResult:
    response: dict[str, Any]
    used_needle: bool
    confidence: float | None
    empty_call: bool
    reasoning: str | None


class NeedleToolPath:
    """Peelable tool-determination path (Maple stays on prose / non-tool turns)."""

    policy_id = "needle2.tool_path"

    def __init__(
        self,
        runtime: NeedleToolRuntime,
        *,
        min_confidence: float = 0.0,
        public_model_id: str = "evprtr",
    ) -> None:
        self.runtime = runtime
        self.min_confidence = min_confidence
        self.public_model_id = public_model_id
        self.last_skip_reason: str | None = None

    def enabled_for(self, request: dict[str, Any]) -> bool:
        return should_route_tools_to_needle(request, enabled=self.runtime.available())

    def handle(self, request: dict[str, Any]) -> ToolPathResult | None:
        """Return a full chat.completion object, or None to fall back to Maple."""
        self.last_skip_reason = None
        if not self.enabled_for(request):
            self.last_skip_reason = "not_enabled"
            return None

        openai_tools = request.get("tools")
        raw_count = len(openai_tools) if isinstance(openai_tools, list) else 0
        needle_tools = openai_tools_to_needle(openai_tools)
        needle_tools = filter_tools_for_choice(needle_tools, request.get("tool_choice"))
        if not needle_tools:
            self.last_skip_reason = (
                f"empty_converted_tools raw_count={raw_count}"
            )
            return None

        query = last_user_text(request.get("messages"))
        if not query:
            self.last_skip_reason = "empty_query"
            return None

        result = self.runtime.complete(query, needle_tools)
        if (
            result.confidence is not None
            and result.confidence < self.min_confidence
            and result.function_calls
        ):
            # Low confidence with a call → let Maple try (escalation).
            self.last_skip_reason = (
                f"low_confidence conf={result.confidence} "
                f"min={self.min_confidence} calls={len(result.function_calls)}"
            )
            return None

        choice = request.get("tool_choice", "auto")
        if result.function_calls:
            message = needle_calls_to_openai_message(result.function_calls, content=None)
            return ToolPathResult(
                response=self._wrap(message, finish_reason="tool_calls"),
                used_needle=True,
                confidence=result.confidence,
                empty_call=False,
                reasoning=result.reasoning,
            )

        # Empty call [] under auto: Needle abstained. Presenting a prose refusal
        # kills the Pi agent loop before Maple can emit structured tool_calls
        # (seen with Pi write/bash schemas → "No tool exists..."). Fall back.
        if choice not in {"required"} and not isinstance(choice, dict):
            reason = (result.reasoning or "").strip().replace("\n", " ")
            self.last_skip_reason = (
                f"empty_call_fallback conf={result.confidence} "
                f"reasoning={reason[:160]!r}"
            )
            return None

        # tool_choice required / forced function: keep an explicit refusal.
        content = _refusal_required(result, needle_tools)
        message = {"role": "assistant", "content": content, "tool_calls": None}
        return ToolPathResult(
            response=self._wrap(message, finish_reason="stop"),
            used_needle=True,
            confidence=result.confidence,
            empty_call=True,
            reasoning=result.reasoning,
        )

    def _wrap(self, message: dict[str, Any], *, finish_reason: str) -> dict[str, Any]:
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.public_model_id,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }


def _refusal_required(result: NeedleCompleteResult, tools: list[dict[str, Any]]) -> str:
    names = ", ".join(t.get("name", "?") for t in tools)
    reason = (result.reasoning or "").strip()
    base = (
        "I could not produce a schema-valid tool call for the available tools "
        f"({names}). The request likely conflicts with required parameters "
        "or cannot be served by these tools."
    )
    if reason:
        return f"{base}\n\nDetails: {reason}"
    return base
