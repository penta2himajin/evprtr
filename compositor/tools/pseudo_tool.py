"""Deterministic parse of Maple pseudo ``<tool_call>`` markup into OpenAI tool_calls."""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

_PSEUDO_BLOCK = re.compile(
    r"<tool_call\b[^>]*>([\s\S]*?)</tool_call>",
    re.IGNORECASE,
)
_NAME_LINE = re.compile(r"(?is)^\s*(?:invoke|call|tool)?\s*([a-zA-Z_][\w.-]*)\s*(?:[\s({].*)?$")


def parse_pseudo_tool_calls(content: str) -> list[dict[str, Any]]:
    """Extract OpenAI-shaped tool_calls from ``<tool_call>...</tool_call>`` blocks.

    Accepts JSON objects (``{"name": "ls", ...}``) or loose ``name\\n{args}`` bodies.
    Returns [] when nothing usable is found.
    """
    text = content or ""
    out: list[dict[str, Any]] = []
    for match in _PSEUDO_BLOCK.finditer(text):
        body = (match.group(1) or "").strip()
        if not body:
            continue
        parsed = _parse_block_body(body)
        if parsed is None:
            continue
        name, args = parsed
        if not name:
            continue
        # Reject shell-ish compound names Maple sometimes invents.
        if any(ch in name for ch in (" ", "/", "\\")):
            continue
        out.append(
            {
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(args, ensure_ascii=False)
                    if not isinstance(args, str)
                    else args,
                },
            }
        )
    return out


def _parse_block_body(body: str) -> tuple[str, dict[str, Any] | str] | None:
    # JSON object first.
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        data = None
    if isinstance(data, dict):
        name = str(data.get("name") or data.get("tool") or "").strip()
        if not name:
            return None
        # HF / DeepGrove shape: {"name", "arguments": {...}}
        if "arguments" in data:
            args = data.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"raw": args}
            if not isinstance(args, dict):
                args = {"value": args}
            return name, args
        # Loose flat object: {"name", "path": ...}
        args = {k: v for k, v in data.items() if k not in {"name", "tool"}}
        return name, args

    # Loose: first line name, rest JSON / key=value.
    lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
    if not lines:
        return None
    m = _NAME_LINE.match(lines[0])
    if not m:
        return None
    name = m.group(1)
    rest = "\n".join(lines[1:]).strip()
    if not rest:
        return name, {}
    try:
        args_obj = json.loads(rest)
        if isinstance(args_obj, dict):
            return name, args_obj
    except json.JSONDecodeError:
        pass
    return name, {"raw": rest}


def apply_pseudo_tool_calls_to_response(
    upstream: dict[str, Any],
    tool_calls: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return a copy of ``upstream`` with structured tool_calls and cleaned content."""
    from compositor.verify.detectors_maple_preview import strip_pseudo_tool_markup

    out = dict(upstream)
    choices = list(out.get("choices") or [])
    if not choices or not isinstance(choices[0], dict):
        return out
    choice = dict(choices[0])
    message = dict(choice.get("message") or {})
    raw_content = message.get("content")
    if isinstance(raw_content, str):
        cleaned = strip_pseudo_tool_markup(raw_content)
        message["content"] = cleaned or None
    message["tool_calls"] = tool_calls
    choice["message"] = message
    choice["finish_reason"] = "tool_calls"
    out["choices"] = [choice, *choices[1:]]
    return out
