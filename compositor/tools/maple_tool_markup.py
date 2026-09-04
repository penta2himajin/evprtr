"""Maple HF/DeepGrove-style ``<tools>`` / ``<tool_call>`` prompt contract.

Harness still speaks OpenAI ``tools`` / ``tool_calls``. The compositor rewrites
the Maple-bound request into markup form, then parses ``<tool_call>`` blocks
back to OpenAI ``tool_calls``. Plain assistant prose stays in ``content``.
"""

from __future__ import annotations

import copy
import json
from typing import Any


_TOOLS_PREAMBLE = (
    "# Tools\n\n"
    "You may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n"
    "<tools>\n"
)

_TOOLS_EPILOGUE = (
    "\n</tools>\n\n"
    "For each function call, return a json object with function name and arguments "
    "within <tool_call></tool_call> XML tags:\n"
    "<tool_call>\n"
    '{"name": <function-name>, "arguments": <args-json-object>}\n'
    "</tool_call>\n"
    "If you can answer without a tool, reply with plain text only "
    "(no <tool_call> tags).\n"
)


def openai_tools_json_lines(tools: list[Any]) -> list[str]:
    """Serialize OpenAI tools entries as JSON lines for ``<tools>``."""
    lines: list[str] = []
    for raw in tools:
        if not isinstance(raw, dict):
            continue
        # Prefer the full OpenAI tool object; fall back to bare function.
        payload: dict[str, Any]
        if raw.get("type") == "function" or "function" in raw:
            payload = raw
        else:
            payload = {"type": "function", "function": raw}
        lines.append(json.dumps(payload, ensure_ascii=False))
    return lines


def tools_system_suffix(tools: list[Any]) -> str:
    """HF chat_template–aligned tools instruction block."""
    body = "\n".join(openai_tools_json_lines(tools))
    return f"{_TOOLS_PREAMBLE}{body}{_TOOLS_EPILOGUE}"


def _format_tool_call_block(name: str, arguments: Any) -> str:
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {"raw": arguments}
    if not isinstance(arguments, dict):
        arguments = {"value": arguments}
    payload = {"name": name, "arguments": arguments}
    return (
        "<tool_call>\n"
        f"{json.dumps(payload, ensure_ascii=False)}\n"
        "</tool_call>"
    )


def rewrite_messages_for_tool_markup(messages: list[Any]) -> list[dict[str, Any]]:
    """Map OpenAI tool history into ``<tool_call>`` / ``<tool_response>`` text."""
    out: list[dict[str, Any]] = []
    for raw in messages:
        if not isinstance(raw, dict):
            continue
        msg = copy.deepcopy(raw)
        role = msg.get("role")
        if role == "assistant" and isinstance(msg.get("tool_calls"), list) and msg["tool_calls"]:
            parts: list[str] = []
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                parts.append(content.strip())
            for call in msg["tool_calls"]:
                if not isinstance(call, dict):
                    continue
                fn = call.get("function") if isinstance(call.get("function"), dict) else call
                if not isinstance(fn, dict):
                    continue
                name = str(fn.get("name") or "").strip()
                if not name:
                    continue
                parts.append(_format_tool_call_block(name, fn.get("arguments", {})))
            msg["content"] = "\n".join(parts) if parts else None
            msg.pop("tool_calls", None)
            out.append(msg)
            continue
        if role == "tool":
            content = msg.get("content")
            if not isinstance(content, str):
                content = "" if content is None else str(content)
            out.append(
                {
                    "role": "user",
                    "content": f"<tool_response>\n{content}\n</tool_response>",
                }
            )
            continue
        msg.pop("tool_calls", None)
        out.append(msg)
    return out


def maple_tool_markup_request(request: dict[str, Any]) -> dict[str, Any]:
    """Prepare a Maple-bound payload using ``<tools>`` / ``<tool_call>`` contract.

    - Injects tools instruction into the system message.
    - Rewrites prior ``tool_calls`` / ``tool`` messages to markup.
    - Removes OpenAI ``tools`` / ``tool_choice`` so upstream is not dual-pathed.
    """
    tools = request.get("tools")
    if not isinstance(tools, list) or not tools:
        return request
    if request.get("tool_choice") == "none":
        return request

    out = copy.deepcopy(request)
    suffix = tools_system_suffix(tools)
    messages = rewrite_messages_for_tool_markup(
        list(out.get("messages") or []) if isinstance(out.get("messages"), list) else []
    )

    if messages and messages[0].get("role") == "system":
        base = messages[0].get("content")
        base_text = base if isinstance(base, str) else ""
        messages[0]["content"] = (
            f"{base_text.rstrip()}\n\n{suffix}" if base_text.strip() else suffix
        )
    else:
        messages.insert(0, {"role": "system", "content": suffix})

    out["messages"] = messages
    out.pop("tools", None)
    out.pop("tool_choice", None)
    return out
