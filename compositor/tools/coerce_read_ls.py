"""Narrow coerce: ``read`` on a directory target → ``ls``.

Maple sometimes emits ``read`` with ``path`` set to a repo root or folder.
That wastes a turn; ``ls`` is the correct tool. Rules stay narrow to avoid
touching extensionless files (``Makefile``, ``LICENSE``, …).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


def _fn_args(call: dict[str, Any]) -> tuple[str, dict[str, Any], dict[str, Any]]:
    fn = call.get("function") if isinstance(call.get("function"), dict) else {}
    name = str(fn.get("name") or call.get("name") or "")
    raw = fn.get("arguments", call.get("arguments"))
    if isinstance(raw, dict):
        return name, dict(raw), fn
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return name, parsed, fn
        except json.JSONDecodeError:
            return name, {}, fn
    return name, {}, fn


def path_looks_like_directory(path: str) -> bool:
    """True for clear directory targets (dot, trailing slash, or absolute dir)."""
    p = (path or "").strip()
    if not p:
        return False
    if p in {".", "./", ".\\"}:
        return True
    if p.endswith(("/", "\\")):
        return True
    try:
        cand = Path(p)
        # Relative FS checks follow compositor cwd, which may not match the
        # harness cwd — only trust absolute paths for is_dir().
        if cand.is_absolute() and cand.is_dir():
            return True
    except OSError:
        return False
    return False


def coerce_read_directory_to_ls(response: dict[str, Any]) -> dict[str, Any]:
    """Rewrite ``read`` tool_calls whose path is a directory into ``ls``."""
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return response
    choice0 = choices[0]
    if not isinstance(choice0, dict):
        return response
    message = choice0.get("message")
    if not isinstance(message, dict):
        return response
    calls = message.get("tool_calls")
    if not isinstance(calls, list) or not calls:
        return response

    changed = False
    out = copy.deepcopy(response)
    out_calls = out["choices"][0]["message"].get("tool_calls") or []
    for call in out_calls:
        if not isinstance(call, dict):
            continue
        name, args, fn = _fn_args(call)
        if name != "read":
            continue
        path = str(args.get("path") or args.get("file") or "").strip()
        if not path_looks_like_directory(path):
            continue
        fn["name"] = "ls"
        # Keep args as JSON string for OpenAI shape consistency.
        fn["arguments"] = json.dumps(args, ensure_ascii=False)
        call["function"] = fn
        changed = True
    if not changed:
        return response
    return out
