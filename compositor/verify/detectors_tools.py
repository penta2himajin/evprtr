"""Detectors for structured tool_calls with unusable arguments.

Maple-Preview (and similar small agent models) sometimes emit a real
``tool_calls`` channel entry whose ``path`` / ``content`` / ``edits`` are
nonsense (e.g. write path=Tower content=0.5, or edit old==new). Approving
those is harmful; stripping and repairing keeps the audit loop intact.
"""

from __future__ import annotations

import json
from typing import Any

from compositor.tools.maple_nl import task_wants_mutation, tool_calls_satisfy_mutation
from compositor.verify.repair import original_user_text, request_has_tools
from compositor.verify.types import Diagnosis

DEGENERATE_TOOL_ARGS_ID = "maple_preview.degenerate_tool_args"
MISSING_MUTATION_ID = "maple_preview.missing_mutation"

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


def _fn_args(call: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    fn = call.get("function") if isinstance(call.get("function"), dict) else {}
    name = str(fn.get("name") or call.get("name") or "")
    raw = fn.get("arguments", call.get("arguments"))
    if isinstance(raw, dict):
        return name, raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return name, parsed
        except json.JSONDecodeError:
            return name, {}
    return name, {}


def _looks_like_path(path: str) -> bool:
    if not path or path in {".", ".."}:
        return False
    if any(sep in path for sep in ("/", "\\")):
        return True
    lower = path.lower()
    return any(lower.endswith(suf) for suf in _CODEISH_SUFFIXES)


class DegenerateToolCallDetector:
    """Reject structured tool_calls whose arguments cannot be a real edit."""

    policy_id = DEGENERATE_TOOL_ARGS_ID

    def diagnose(
        self,
        message: dict[str, Any],
        *,
        request: dict[str, Any] | None = None,
    ) -> Diagnosis | None:
        del request
        calls = message.get("tool_calls")
        if not isinstance(calls, list) or not calls:
            return None

        for idx, call in enumerate(calls):
            if not isinstance(call, dict):
                continue
            name, args = _fn_args(call)
            if name == "write":
                hit = self._write_problem(args)
                if hit is not None:
                    return Diagnosis(
                        policy_id=self.policy_id,
                        field="tool_calls",
                        kind="degenerate_tool_args",
                        onset=idx,
                        detail={"tool": "write", **hit},
                    )
            elif name == "edit":
                hit = self._edit_problem(args)
                if hit is not None:
                    return Diagnosis(
                        policy_id=self.policy_id,
                        field="tool_calls",
                        kind="degenerate_tool_args",
                        onset=idx,
                        detail={"tool": "edit", **hit},
                    )
        return None

    def _write_problem(self, args: dict[str, Any]) -> dict[str, Any] | None:
        path = str(args.get("path") or "").strip()
        content = str(args.get("content") or "")
        if not _looks_like_path(path):
            return {"reason": "path_not_filelike", "path": path, "content_len": len(content)}
        if content.strip() == path.strip():
            return {"reason": "content_equals_path", "path": path, "content_len": len(content)}
        lower_path = path.lower().replace("\\", "/")
        if lower_path.endswith("cargo.toml"):
            if "[package]" not in content and "axum" not in content and len(content) < 80:
                return {
                    "reason": "cargo_toml_too_thin",
                    "path": path,
                    "content_len": len(content),
                }
        if any(lower_path.endswith(suf) for suf in (".rs", ".toml", ".py", ".ts")):
            if len(content.strip()) < 12:
                return {
                    "reason": "source_content_too_short",
                    "path": path,
                    "content_len": len(content),
                }
        return None

    def _edit_problem(self, args: dict[str, Any]) -> dict[str, Any] | None:
        path = str(args.get("path") or "").strip()
        if path and not _looks_like_path(path):
            return {"reason": "path_not_filelike", "path": path}
        edits = args.get("edits")
        if not isinstance(edits, list) or not edits:
            return {"reason": "empty_edits", "path": path}
        usable = 0
        for edit in edits:
            if not isinstance(edit, dict):
                continue
            old = str(edit.get("oldText") or edit.get("old_string") or "")
            new = str(edit.get("newText") or edit.get("new_string") or "")
            if not old and not new:
                return {"reason": "empty_edit_texts", "path": path}
            if old and old == new:
                return {"reason": "noop_edit", "path": path, "old_len": len(old)}
            # Prompt-echo / meta text often appears as invented oldText (live tinyserve).
            low = old.lower()
            if any(
                tok in low
                for tok in (
                    "non-empty oldtext",
                    "oldtext and newtext",
                    "mutation tool",
                    "use the edit",
                    "do not only",
                )
            ):
                return {"reason": "invented_old_text", "path": path}
            if old and len(old) < 80 and "tokio {version" in old.replace(" ", ""):
                # Seen live: invented oldText that never exists in the file.
                return {"reason": "invented_old_text", "path": path}
            usable += 1
        if usable == 0:
            return {"reason": "empty_edit_texts", "path": path}
        return None


def _offered_mutation_tools(request: dict[str, Any] | None) -> bool:
    tools = (request or {}).get("tools")
    if not isinstance(tools, list):
        return False
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        name = str((fn or {}).get("name") or "")
        if name in {"write", "edit", "bash", "powershell"}:
            return True
    return False


class MissingMutationDetector:
    """Task needs write/edit but the model stopped with only read/prose."""

    policy_id = MISSING_MUTATION_ID

    def diagnose(
        self,
        message: dict[str, Any],
        *,
        request: dict[str, Any] | None = None,
    ) -> Diagnosis | None:
        if not request_has_tools(request or {}):
            return None
        if not _offered_mutation_tools(request):
            return None
        task = original_user_text(request or {})
        if not task_wants_mutation(task):
            return None
        wrapped = {"choices": [{"message": message}]}
        if tool_calls_satisfy_mutation(wrapped):
            return None
        calls = message.get("tool_calls") or []
        names = []
        for call in calls:
            if isinstance(call, dict):
                fn = call.get("function") if isinstance(call.get("function"), dict) else {}
                names.append(str(fn.get("name") or ""))
        return Diagnosis(
            policy_id=self.policy_id,
            field="tool_calls",
            kind="missing_mutation",
            onset=0,
            detail={
                "reason": "read_or_prose_without_write",
                "tool_names": names,
                "task_head": task[:160],
            },
        )
