"""Derive oMLX ``structured_outputs`` from OpenAI ``tools`` (Needle-off path).

When Maple is given ``tools`` alone, oMLX may not install a tool-name grammar.
We inject a JSON schema (``oneOf`` per tool) via ``structured_outputs``. Live
oMLX then returns the tool as assistant **content** JSON; ``promote_tool_json_content``
lifts that into OpenAI ``tool_calls``.

The schema also includes a ``{"message": "..."}`` branch so multi-turn agent loops
can emit a prose answer without being forced into another tool call forever.
"""

from __future__ import annotations

import copy
import json
import uuid
from typing import Any


def _tool_entries(tools: list[Any]) -> list[tuple[str, dict[str, Any]]]:
    out: list[tuple[str, dict[str, Any]]] = []
    for raw in tools:
        if not isinstance(raw, dict):
            continue
        fn = raw.get("function") if isinstance(raw.get("function"), dict) else raw
        if not isinstance(fn, dict):
            continue
        name = str(fn.get("name") or "").strip()
        if not name:
            continue
        params = fn.get("parameters")
        if not isinstance(params, dict):
            params = {"type": "object", "properties": {}}
        out.append((name, params))
    return out


def _message_branch() -> dict[str, Any]:
    """Prose-stop alternative to a tool call (agent-loop escape hatch)."""
    return {
        "type": "object",
        "properties": {
            "message": {"type": "string"},
        },
        "required": ["message"],
        "additionalProperties": False,
    }


def _tool_branch(name: str, params: dict[str, Any]) -> dict[str, Any]:
    args_schema = copy.deepcopy(params)
    if args_schema.get("type") is None:
        args_schema["type"] = "object"
    return {
        "type": "object",
        "properties": {
            "name": {"const": name},
            "arguments": args_schema,
        },
        "required": ["name", "arguments"],
        "additionalProperties": False,
    }


def json_schema_from_openai_tools(tools: list[Any]) -> dict[str, Any]:
    """Build JSON Schema ``oneOf``: each tool call shape, plus a message branch."""
    branches: list[dict[str, Any]] = [
        _tool_branch(name, params) for name, params in _tool_entries(tools)
    ]
    branches.append(_message_branch())
    if len(branches) == 1:
        return branches[0]
    return {"oneOf": branches}


def attach_tools_structured_outputs(request: dict[str, Any]) -> dict[str, Any]:
    """Return a copy with ``structured_outputs.json`` from tools when appropriate.

    No-op (returns the same mapping conceptually without injection) when:
    - no tools
    - ``tool_choice`` is ``none``
    - harness already set ``structured_outputs`` or ``guided_grammar``
    """
    tools = request.get("tools")
    if not isinstance(tools, list) or not tools:
        return request
    if request.get("tool_choice") == "none":
        return request
    if request.get("structured_outputs") is not None:
        return request
    if request.get("guided_grammar"):
        return request

    schema = json_schema_from_openai_tools(tools)
    out = copy.deepcopy(request)
    out["structured_outputs"] = {"json": schema}
    # Constrained decoding alone does not teach the escape hatch; hint once.
    hint = (
        "When calling a tool, output JSON "
        '{"name":"<tool>","arguments":{...}}. '
        'When finished answering the user (no more tools), output JSON '
        '{"message":"<your reply>"}.'
    )
    messages = out.get("messages")
    if isinstance(messages, list):
        messages = list(messages)
        messages.insert(0, {"role": "system", "content": hint})
        out["messages"] = messages
    return out


def _parse_json_object(content: str) -> dict[str, Any] | None:
    text = (content or "").strip()
    if not text:
        return None
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            obj = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return obj if isinstance(obj, dict) else None


def _normalize_args(args: Any) -> dict[str, Any]:
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return {"raw": args}
    if not isinstance(args, dict):
        return {"value": args}
    return args


def promote_tool_json_content(response: dict[str, Any]) -> dict[str, Any]:
    """Promote tool JSON → ``tool_calls``, or ``{"message":...}`` → prose content."""
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return response
    choice0 = choices[0]
    if not isinstance(choice0, dict):
        return response
    message = choice0.get("message")
    if not isinstance(message, dict):
        return response
    existing = message.get("tool_calls")
    if isinstance(existing, list) and existing:
        return response
    content = message.get("content")
    if not isinstance(content, str):
        return response
    parsed = _parse_json_object(content)
    if parsed is None:
        return response

    out = copy.deepcopy(response)
    msg = out["choices"][0]["message"]

    # Final prose escape: {"message": "..."}
    if "message" in parsed and "name" not in parsed:
        text = parsed.get("message")
        if not isinstance(text, str):
            text = json.dumps(text, ensure_ascii=False)
        msg["content"] = text
        msg["tool_calls"] = None
        out["choices"][0]["finish_reason"] = "stop"
        return out

    name = parsed.get("name")
    if not isinstance(name, str) or not name.strip():
        return response

    call_id = f"call_{uuid.uuid4().hex[:8]}"
    msg["tool_calls"] = [
        {
            "id": call_id,
            "type": "function",
            "function": {
                "name": name.strip(),
                "arguments": json.dumps(
                    _normalize_args(parsed.get("arguments", {})),
                    ensure_ascii=False,
                ),
            },
        }
    ]
    msg["content"] = None
    out["choices"][0]["finish_reason"] = "tool_calls"
    return out
