"""Classify tool calls as safe (passthrough) vs side-effect (buffer)."""

from __future__ import annotations

from typing import Any

# Names match common harness tools (Pi, OpenCode-style) and browser/search extras.
SIDE_EFFECT_TOOLS: frozenset[str] = frozenset(
    {
        "write",
        "edit",
        "apply_patch",
        "bash",
        "shell",
        "terminal",
        "run_terminal_cmd",
        "execute",
        "web_search",
        "web_fetch",
        "WebSearch",
        "WebFetch",
        "browser",
        "browser_click",
        "browser_navigate",
        "browser_type",
        "mcp_web_fetch",
        "search",
    }
)

SAFE_TOOLS: frozenset[str] = frozenset(
    {
        "read",
        "grep",
        "glob",
        "list_dir",
        "search_replace_preview",
    }
)


def tool_name(tool_call: dict[str, Any]) -> str:
    if not isinstance(tool_call, dict):
        return ""
    fn = tool_call.get("function")
    if isinstance(fn, dict) and fn.get("name"):
        return str(fn["name"])
    return str(tool_call.get("name") or "")


def is_side_effect_tool(name: str) -> bool:
    if not name:
        return False
    if name in SAFE_TOOLS:
        return False
    if name in SIDE_EFFECT_TOOLS:
        return True
    lower = name.lower()
    # Conservative: unknown write/shell-ish names are buffered.
    for hint in ("write", "edit", "bash", "shell", "exec", "search", "fetch", "browser"):
        if hint in lower:
            return True
    return False


def split_tool_calls(
    tool_calls: list[dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (safe_passthrough, buffered_side_effects)."""
    safe: list[dict[str, Any]] = []
    buffered: list[dict[str, Any]] = []
    for call in tool_calls or []:
        if is_side_effect_tool(tool_name(call)):
            buffered.append(call)
        else:
            safe.append(call)
    return safe, buffered
