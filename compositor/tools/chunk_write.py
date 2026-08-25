"""Split large write payloads; Needle (or deterministic fallback) applies chunks.

Pi has no write_partial tool, so chunks become a write of part1+sentinel, then
edit steps that replace the sentinel until the final chunk clears it.
"""

from __future__ import annotations

import json
from typing import Any

CHUNK_SENTINEL = "\n/*EVPRTR_CHUNK_MORE*/\n"


def split_content(content: str, *, max_chars: int = 900) -> list[str]:
    """Split on line boundaries when possible."""
    text = content if isinstance(content, str) else str(content)
    if max_chars <= 0 or len(text) <= max_chars:
        return [text]
    lines = text.splitlines(keepends=True)
    chunks: list[str] = []
    buf = ""
    for line in lines:
        if buf and len(buf) + len(line) > max_chars:
            chunks.append(buf)
            buf = line
        else:
            buf += line
        if len(buf) > max_chars:
            # Single overlong line: hard-split.
            while len(buf) > max_chars:
                chunks.append(buf[:max_chars])
                buf = buf[max_chars:]
    if buf:
        chunks.append(buf)
    return chunks or [text]


def planned_tool_calls(path: str, chunks: list[str]) -> list[dict[str, Any]]:
    """Deterministic OpenAI-shaped tool_calls for chunked write/edit apply."""
    if not chunks:
        return []
    if len(chunks) == 1:
        return [_write_call(path, chunks[0])]

    calls: list[dict[str, Any]] = [
        _write_call(path, chunks[0] + CHUNK_SENTINEL)
    ]
    for i, chunk in enumerate(chunks[1:], start=1):
        last = i == len(chunks) - 1
        new_text = chunk if last else chunk + CHUNK_SENTINEL
        calls.append(
            _edit_call(
                path,
                old_text=CHUNK_SENTINEL,
                new_text=new_text,
            )
        )
    return calls


def needle_chunk_query(path: str, chunk: str, *, index: int, total: int) -> str:
    if total == 1:
        return (
            f"Call the write tool once. path={path!r}. "
            f"content must be exactly:\n{chunk}"
        )
    if index == 0:
        return (
            f"Call write. path={path!r}. content must be exactly this first "
            f"chunk ({index + 1}/{total}) followed by the sentinel "
            f"{CHUNK_SENTINEL!r}:\n{chunk}{CHUNK_SENTINEL}"
        )
    last = index == total - 1
    new_text = chunk if last else chunk + CHUNK_SENTINEL
    return (
        f"Call edit on path={path!r}. Replace oldText={CHUNK_SENTINEL!r} "
        f"with newText exactly equal to chunk {index + 1}/{total}:\n{new_text}"
    )


def _write_call(path: str, content: str) -> dict[str, Any]:
    return {
        "id": f"call_chunk_w_{abs(hash((path, content[:32]))) % 10**10}",
        "type": "function",
        "function": {
            "name": "write",
            "arguments": json.dumps(
                {"path": path, "content": content}, ensure_ascii=False
            ),
        },
    }


def _edit_call(path: str, *, old_text: str, new_text: str) -> dict[str, Any]:
    return {
        "id": f"call_chunk_e_{abs(hash((path, old_text, new_text[:32]))) % 10**10}",
        "type": "function",
        "function": {
            "name": "edit",
            "arguments": json.dumps(
                {
                    "path": path,
                    "edits": [{"oldText": old_text, "newText": new_text}],
                },
                ensure_ascii=False,
            ),
        },
    }
