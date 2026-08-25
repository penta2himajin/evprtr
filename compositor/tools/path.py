"""Tool path: Needle determines calls; compositor presents OpenAI-compatible shape."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any

from compositor.runtimes.needle import NeedleCompleteResult, NeedleToolRuntime
from compositor.tools.chunk_write import (
    needle_chunk_query,
    planned_tool_calls,
    split_content,
)
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
    via: str = "needle_direct"


class NeedleToolPath:
    """Peelable tool-determination path (Maple stays on prose / non-tool turns)."""

    policy_id = "needle2.tool_path"

    def __init__(
        self,
        runtime: NeedleToolRuntime,
        *,
        min_confidence: float = 0.0,
        public_model_id: str = "evprtr",
        chunk_chars: int = 900,
        prefer_chunk_writes: bool = True,
    ) -> None:
        self.runtime = runtime
        self.min_confidence = min_confidence
        self.public_model_id = public_model_id
        self.chunk_chars = max(200, chunk_chars)
        self.prefer_chunk_writes = prefer_chunk_writes
        self.last_skip_reason: str | None = None

    def enabled_for(self, request: dict[str, Any]) -> bool:
        return should_route_tools_to_needle(request, enabled=self.runtime.available())

    def handle(self, request: dict[str, Any]) -> ToolPathResult | None:
        """Needle on the raw user text (legacy / fallback path)."""
        query = last_user_text(request.get("messages"))
        if not query:
            self.last_skip_reason = "empty_query"
            return None
        return self.handle_instruction(query, request, via="needle_direct")

    def handle_instruction(
        self,
        instruction: str,
        request: dict[str, Any],
        *,
        via: str = "needle_from_maple_nl",
    ) -> ToolPathResult | None:
        """Structure a natural-language instruction into OpenAI tool_calls."""
        self.last_skip_reason = None
        if not self.enabled_for(request):
            self.last_skip_reason = "not_enabled"
            return None

        needle_tools = self._needle_tools(request)
        if not needle_tools:
            return None

        text = (instruction or "").strip()
        if not text:
            self.last_skip_reason = "empty_instruction"
            return None

        result = self.runtime.complete(text, needle_tools)
        return self._result_from_complete(result, request, needle_tools, via=via)

    def apply_chunked_file(
        self,
        path: str,
        content: str,
        request: dict[str, Any],
        *,
        via: str = "needle_chunk_write",
    ) -> ToolPathResult | None:
        """Split ``content`` and apply via Needle per chunk, else deterministic plan."""
        self.last_skip_reason = None
        if not self.enabled_for(request):
            self.last_skip_reason = "not_enabled"
            return None

        chunks = split_content(content, max_chars=self.chunk_chars)
        needle_tools = [
            t
            for t in self._needle_tools(request)
            if t.get("name") in {"write", "edit"}
        ]
        if not needle_tools:
            self.last_skip_reason = "no_write_edit_tools"
            return None

        if not self.prefer_chunk_writes or len(chunks) == 1:
            # Single shot: ask Needle to write the whole file; fall back to plan.
            query = needle_chunk_query(path, content, index=0, total=1)
            result = self.runtime.complete(query, needle_tools)
            shaped = self._result_from_complete(
                result, request, needle_tools, via=via, allow_empty_fallback=False
            )
            if shaped is not None and not shaped.empty_call:
                return shaped
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": planned_tool_calls(path, [content]),
            }
            return ToolPathResult(
                response=self._wrap(message, finish_reason="tool_calls"),
                used_needle=False,
                confidence=None,
                empty_call=False,
                reasoning="deterministic_single_write",
                via=f"{via}_deterministic",
            )

        # Prefer deterministic multi-call plan for reliability; optionally probe
        # Needle on chunk 0 to confirm it understands write.
        planned = planned_tool_calls(path, chunks)
        probe = self.runtime.complete(
            needle_chunk_query(path, chunks[0], index=0, total=len(chunks)),
            [t for t in needle_tools if t.get("name") == "write"] or needle_tools,
        )
        used_needle = bool(probe.function_calls)
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": planned,
        }
        return ToolPathResult(
            response=self._wrap(message, finish_reason="tool_calls"),
            used_needle=used_needle,
            confidence=probe.confidence,
            empty_call=False,
            reasoning=(
                f"chunked_write parts={len(chunks)} "
                f"needle_probe_calls={len(probe.function_calls or [])}"
            ),
            via=via if used_needle else f"{via}_deterministic",
        )

    def _needle_tools(self, request: dict[str, Any]) -> list[dict[str, Any]]:
        openai_tools = request.get("tools")
        raw_count = len(openai_tools) if isinstance(openai_tools, list) else 0
        needle_tools = openai_tools_to_needle(openai_tools)
        needle_tools = filter_tools_for_choice(needle_tools, request.get("tool_choice"))
        if not needle_tools:
            self.last_skip_reason = f"empty_converted_tools raw_count={raw_count}"
        return needle_tools

    def _result_from_complete(
        self,
        result: NeedleCompleteResult,
        request: dict[str, Any],
        needle_tools: list[dict[str, Any]],
        *,
        via: str,
        allow_empty_fallback: bool = True,
    ) -> ToolPathResult | None:
        if (
            result.confidence is not None
            and result.confidence < self.min_confidence
            and result.function_calls
        ):
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
                via=via,
            )

        if allow_empty_fallback and choice not in {"required"} and not isinstance(
            choice, dict
        ):
            reason = (result.reasoning or "").strip().replace("\n", " ")
            self.last_skip_reason = (
                f"empty_call_fallback conf={result.confidence} "
                f"reasoning={reason[:160]!r}"
            )
            return None

        content = _refusal_required(result, needle_tools)
        message = {"role": "assistant", "content": content, "tool_calls": None}
        return ToolPathResult(
            response=self._wrap(message, finish_reason="stop"),
            used_needle=True,
            confidence=result.confidence,
            empty_call=True,
            reasoning=result.reasoning,
            via=via,
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
