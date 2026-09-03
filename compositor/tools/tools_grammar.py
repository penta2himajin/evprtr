"""Derive oMLX ``structured_outputs`` from OpenAI ``tools`` (Needle-off path).

When Maple is given ``tools`` alone, oMLX may not install a tool-name grammar.
We inject a JSON schema (``oneOf`` per tool) via ``structured_outputs``. Live
oMLX then returns the tool as assistant **content** JSON; ``promote_tool_json_content``
lifts that into OpenAI ``tool_calls``.
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


def json_schema_from_openai_tools(tools: list[Any]) -> dict[str, Any]:
    """Build a JSON Schema ``oneOf`` constraining ``name`` + ``arguments``."""
    branches: list[dict[str, Any]] = []
    for name, params in _tool_entries(tools):
        args_schema = copy.deepcopy(params)
        if args_schema.get("type") is None:
            args_schema["type"] = "object"
        branches.append(
            {
                "type": "object",
                "properties": {
                    "name": {"const": name},
                    "arguments": args_schema,
                },
                "required": ["name", "arguments"],
                "additionalProperties": False,
            }
        )
    if not branches:
        return {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "arguments": {"type": "object"},
            },
            "required": ["name", "arguments"],
        }
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
    return out


def _extract_tool_obj(content: str) -> dict[str, Any] | None:
    text = (content or "").strip()
    if not text:
        return None
    # Fenced ```json ... ```
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
        # Tolerate leading/trailing prose by finding first {...}
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            obj = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    if not isinstance(obj, dict):
        return None
    name = obj.get("name")
    if not isinstance(name, str) or not name.strip():
        return None
    args = obj.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {"raw": args}
    if not isinstance(args, dict):
        args = {"value": args}
    return {"name": name.strip(), "arguments": args}


def promote_tool_json_content(response: dict[str, Any]) -> dict[str, Any]:
    """If assistant content is tool JSON and there are no tool_calls, promote it."""
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
    parsed = _extract_tool_obj(content)
    if parsed is None:
        return response

    out = copy.deepcopy(response)
    msg = out["choices"][0]["message"]
    call_id = f"call_{uuid.uuid4().hex[:8]}"
    msg["tool_calls"] = [
        {
            "id": call_id,
            "type": "function",
            "function": {
                "name": parsed["name"],
                "arguments": json.dumps(parsed["arguments"], ensure_ascii=False),
            },
        }
    ]
    msg["content"] = None
    out["choices"][0]["finish_reason"] = "tool_calls"
    return out
