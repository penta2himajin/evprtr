"""Convert between OpenAI tool schemas and Needle schemas / responses."""

from __future__ import annotations

import json
import uuid
from typing import Any


def openai_tools_to_needle(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """OpenAI `tools: [{type:function,function:{name,description,parameters}}]` → Needle list."""
    out: list[dict[str, Any]] = []
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") and tool.get("type") != "function":
            continue
        fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        if not isinstance(fn, dict) or not fn.get("name"):
            continue
        entry: dict[str, Any] = {
            "name": fn["name"],
            "description": fn.get("description") or "",
            "parameters": fn.get("parameters")
            or {"type": "object", "properties": {}, "required": []},
        }
        out.append(entry)
    return out


def filter_tools_for_choice(
    tools: list[dict[str, Any]],
    tool_choice: str | dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """When tool_choice forces a function, expose only that tool to Needle."""
    if not isinstance(tool_choice, dict):
        return tools
    if tool_choice.get("type") != "function":
        return tools
    name = (tool_choice.get("function") or {}).get("name")
    if not name:
        return tools
    return [t for t in tools if t.get("name") == name]


def needle_calls_to_openai_message(
    function_calls: list[dict[str, Any]],
    *,
    content: str | None = None,
) -> dict[str, Any]:
    """Build an assistant message with structured tool_calls."""
    tool_calls = []
    for call in function_calls:
        name = call.get("name")
        args = call.get("arguments")
        if not name:
            continue
        if isinstance(args, str):
            args_s = args
        else:
            args_s = json.dumps(args or {}, ensure_ascii=False)
        tool_calls.append(
            {
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {"name": name, "arguments": args_s},
            }
        )
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": tool_calls,
    }


def last_user_text(messages: list[dict[str, Any]] | None) -> str:
    parts: list[str] = []
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            parts.append(content.strip())
    return "\n\n".join(parts) if parts else ""


def should_route_tools_to_needle(
    request: dict[str, Any],
    *,
    enabled: bool,
) -> bool:
    """Needle owns tool determination when tools are offered and not disabled."""
    if not enabled:
        return False
    tools = request.get("tools")
    if not tools:
        return False
    choice = request.get("tool_choice", "auto")
    if choice == "none":
        return False
    return True
