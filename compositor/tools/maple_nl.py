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
        "- At most 8 short lines.\n"
        "- Do not emit <tool_call> markup or JSON tool calls.\n"
        "- Name the tool (write, edit, read, bash, …), paths, and exact file "
        "contents when writing files.\n"
        "- For small files, put the full body in a markdown fence labeled with "
        "the path (example: ```live-nl-smoke.txt).\n"
        "- Never repeat tokens or pad with filler."
    )
    messages = list(payload.get("messages") or [])
    payload["messages"] = [{"role": "system", "content": system}, *messages]
    payload["stream"] = False
    # Keep NL short — live Maple otherwise dumped 10k+ repeated tokens.
    payload["max_tokens"] = 384
    payload["temperature"] = 0.1
    payload["frequency_penalty"] = 1.0
    payload["presence_penalty"] = 0.8
    payload["repeat_penalty"] = 1.4
    return payload


def prepare_needle_instruction(maple_nl: str, *, user_task: str, limit: int = 1200) -> str:
    """Clamp Maple NL; if unusable, fall back to a compact user-task brief."""
    text = (maple_nl or "").strip()
    fences = extract_fenced_files(text)
    if fences:
        # Keep path fences + a short head so Needle sees the verb.
        head = text[:240].strip()
        parts = [head] if head else []
        for path, body in fences[:2]:
            parts.append(f"Write path={path} with content:\n```{path}\n{body}```")
        text = "\n\n".join(parts)
    if len(text) > limit or _looks_degenerate_nl(text):
        task = (user_task or "").strip()
        return (
            "Convert this coding-agent task into one or more structured tool calls. "
            "Prefer write/edit with full file content when creating or updating files.\n\n"
            f"Task:\n{task[:limit]}"
        )
    return text[:limit]


def _looks_degenerate_nl(text: str) -> bool:
    if len(text) < 8:
        return True
    # Heavy character-run / repeated motif — Maple NL collapse.
    if len(set(text[:200])) < 8 and len(text) > 80:
        return True
    sample = text[:400]
    for n in (3, 4, 5):
        if len(sample) < n * 6:
            continue
        gram = sample[:n]
        if sample.count(gram) >= 8:
            return True
    return False


_CREATE_FILE = re.compile(
    r"(?is)create\s+file\s+(?P<path>[\w./\\-]+\.\w+)\s+with\s+(?:exactly\s+)?(?:this\s+)?content\s*:?\s*(?P<body>.+?)(?:\n\s*use\s+the\s+write|\Z)"
)


def user_task_instruction(user_task: str, *, limit: int = 1200) -> str:
    """Render a Needle-friendly imperative brief from the user task.

    Live probes showed Needle reliably fills ``write`` for
    ``Call write. path=... content must be exactly: ...`` but often abstains
    or picks ``read`` on soft “convert this task” phrasing.
    """
    task = (user_task or "").strip()
    m = _CREATE_FILE.search(task)
    if m:
        path = m.group("path").strip()
        body = m.group("body").strip()
        # Drop trailing instructional lines from the capture.
        body = re.split(r"(?i)\n\s*use the write", body)[0].strip()
        return f'Call write. path="{path}". content must be exactly:\n{body}'
    return (
        "Call the needed tools now. For new or overwritten files call write "
        "with path and the full content. Do not call read unless the task "
        "requires inspecting an existing file first.\n\n"
        f"Task:\n{task[:limit]}"
    )


def task_wants_mutation(user_task: str) -> bool:
    t = (user_task or "").lower()
    keys = (
        "write",
        "create file",
        "create a file",
        "edit",
        "update file",
        "save ",
        "add dependency",
        "implement",
        "patch",
    )
    return any(k in t for k in keys)


def tool_calls_satisfy_mutation(response: dict[str, Any]) -> bool:
    choices = response.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return False
    message = choices[0].get("message") or {}
    calls = message.get("tool_calls") or []
    mutate = {"write", "edit", "bash", "powershell"}
    for call in calls:
        if not isinstance(call, dict):
            continue
        fn = call.get("function") if isinstance(call.get("function"), dict) else {}
        name = str(fn.get("name") or "")
        if name in mutate:
            return True
    return False


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
