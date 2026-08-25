"""Fake SSE streaming: run a full completion, then emit OpenAI chat chunks.

Harnesses (e.g. Pi) often send ``stream=true`` and have no switch to disable it.
True token streaming through verify/Needle is still future work; this shim unblocks
the agent loop while keeping compositor logic non-streaming.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any


def completion_to_sse_chunks(response: dict[str, Any]) -> list[str]:
    """Return SSE ``data:`` lines (without trailing blank line separators)."""
    model = str(response.get("model") or "evprtr")
    created = int(response.get("created") or time.time())
    cid = str(response.get("id") or f"chatcmpl-evprtr-{created}")
    choices = response.get("choices") or []
    message: dict[str, Any] = {}
    finish_reason = "stop"
    if choices and isinstance(choices[0], dict):
        message = choices[0].get("message") or {}
        finish_reason = str(choices[0].get("finish_reason") or "stop")
        if not isinstance(message, dict):
            message = {}

    delta: dict[str, Any] = {"role": "assistant"}
    content = message.get("content")
    if content is not None:
        delta["content"] = content
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    if reasoning is not None:
        delta["reasoning_content"] = reasoning
    tool_calls = message.get("tool_calls")
    if tool_calls:
        # Emit as a single chunk with full tool_calls (arguments already complete).
        delta["tool_calls"] = []
        for i, call in enumerate(tool_calls):
            if not isinstance(call, dict):
                continue
            item = dict(call)
            item.setdefault("index", i)
            delta["tool_calls"].append(item)

    first = {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
    }
    last = {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
    }
    return [
        f"data: {json.dumps(first, ensure_ascii=False)}",
        f"data: {json.dumps(last, ensure_ascii=False)}",
        "data: [DONE]",
    ]


async def iter_sse_from_completion(response: dict[str, Any]) -> AsyncIterator[str]:
    for line in completion_to_sse_chunks(response):
        yield line + "\n\n"
