"""Narrow coerce: ``read``↔``ls`` when the path target is obviously wrong.

- ``read`` on a directory → ``ls``
- ``ls`` on a file → ``read``

Rules stay narrow to avoid touching ambiguous extensionless paths
(``Makefile``, ``LICENSE``, …) unless an absolute ``is_file()`` check confirms.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

_CODEISH_SUFFIXES = (
    ".toml",
    ".rs",
    ".py",
    ".ts",
    ".js",
    ".json",
    ".md",
    ".txt",
    ".yml",
    ".yaml",
)


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


def path_looks_like_file(path: str) -> bool:
    """True for clear file targets (code-ish suffix or absolute is_file)."""
    p = (path or "").strip()
    if not p or path_looks_like_directory(p):
        return False
    lower = p.lower().replace("\\", "/")
    if any(lower.endswith(suf) for suf in _CODEISH_SUFFIXES):
        return True
    try:
        cand = Path(p)
        if cand.is_absolute() and cand.is_file():
            return True
    except OSError:
        return False
    return False


def _rewrite_tool_name(
    response: dict[str, Any],
    *,
    from_name: str,
    to_name: str,
    path_ok,
) -> dict[str, Any]:
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
        if name != from_name:
            continue
        path = str(args.get("path") or args.get("file") or "").strip()
        if not path_ok(path):
            continue
        fn["name"] = to_name
        fn["arguments"] = json.dumps(args, ensure_ascii=False)
        call["function"] = fn
        changed = True
    if not changed:
        return response
    return out


def coerce_read_directory_to_ls(response: dict[str, Any]) -> dict[str, Any]:
    """Rewrite ``read`` tool_calls whose path is a directory into ``ls``."""
    return _rewrite_tool_name(
        response,
        from_name="read",
        to_name="ls",
        path_ok=path_looks_like_directory,
    )


def coerce_ls_file_to_read(response: dict[str, Any]) -> dict[str, Any]:
    """Rewrite ``ls`` tool_calls whose path is a file into ``read``."""
    return _rewrite_tool_name(
        response,
        from_name="ls",
        to_name="read",
        path_ok=path_looks_like_file,
    )
