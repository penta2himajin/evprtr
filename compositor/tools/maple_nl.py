"""Maple natural-language pass before Needle structures tool calls."""

from __future__ import annotations

import copy
import re
from typing import Any

_FENCE = re.compile(r"```(?:[\w.+-]*)\n([\s\S]*?)```", re.MULTILINE)
_PATH_HINT = re.compile(
    r"(?i)(?:write|create|update|save|path)\s*[:=]\s*[`\"]?([\w./\\-]+\.\w+)"
)


def maple_nl_request(request: dict[str, Any]) -> dict[str, Any]:
    """Ask Maple for natural-language tool instructions only (no tool channel)."""
    payload = copy.deepcopy(request)
    payload.pop("tools", None)
    payload["tool_choice"] = "none"
    system = (
        "You are a coding-agent planner. Describe the next tool actions in clear "
        "natural language for a structured tool engine.\n"
        "Rules:\n"
        "- Do not emit <tool_call> markup or JSON tool calls.\n"
        "- Name the tool (write, edit, read, bash, …), paths, and exact file "
        "contents when writing files.\n"
        "- Prefer one write with full file content inside a markdown code fence "
        "labeled with the path, e.g. ```Cargo.toml then the body.\n"
        "- Keep reasoning short; avoid runaway repetition."
    )
    messages = list(payload.get("messages") or [])
    payload["messages"] = [{"role": "system", "content": system}, *messages]
    payload["stream"] = False
    # Room for a short plan + one fenced file body.
    payload["max_tokens"] = min(int(payload.get("max_tokens") or 2048), 2048)
    payload["temperature"] = 0.2
    return payload


def assistant_text(upstream: dict[str, Any]) -> str:
    choices = upstream.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                t = block.get("text")
                if isinstance(t, str) and t.strip():
                    parts.append(t.strip())
        return "\n".join(parts)
    return ""


def extract_fenced_files(text: str) -> list[tuple[str, str]]:
    """Return (path, content) pairs from fenced blocks when a path is hinted."""
    out: list[tuple[str, str]] = []
    for match in _FENCE.finditer(text or ""):
        body = match.group(1)
        if not body.strip():
            continue
        # Prefer info-string path: ```Cargo.toml / ```path=src/main.rs
        start = match.start()
        path = ""
        info = re.search(r"```([^\n`]+)", text[start : start + 80])
        if info:
            token = info.group(1).strip()
            if "/" in token or "\\" in token or "." in token:
                path = token.split()[0].removeprefix("path=").strip("`\"'")
        if not path:
            hint = _PATH_HINT.search(text[max(0, start - 200) : start])
            if hint:
                path = hint.group(1)
        if not path:
            # Skip anonymous fences for chunk routing; Needle still sees full NL.
            continue
        out.append((path, body if body.endswith("\n") else body + "\n"))
    return out


def correction_instruction(
    *,
    task: str,
    tool: str | None,
    reason: str | None,
    path: str | None,
    failed_preview: str,
) -> str:
    return (
        "Previous structured tool call was unusable and must be replaced.\n"
        f"Failed tool={tool!r} reason={reason!r} path={path!r}.\n"
        f"Failed preview:\n{failed_preview[:500]}\n\n"
        "Emit one correct tool call for the coding task below. "
        "For file writes, include the full file content.\n\n"
        f"Task:\n{task}"
    )
