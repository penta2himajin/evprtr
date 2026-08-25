"""Detect identical gated mutations to stop rewrite / reject loops early."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal


Action = Literal["wait", "auto_reject", "abort"]


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


def fingerprint_write(path: str, content: str) -> str:
    return f"write:{path}:{_sha(content)}"


def fingerprint_edit(path: str, edits: Any) -> str:
    return f"edit:{path}:{_sha(json.dumps(edits, ensure_ascii=False, sort_keys=True))}"


def fingerprint_shell(tool: str, command: str) -> str:
    return f"{tool}:{_sha(command)}"


def fingerprint_from_input(tool_name: str, input_obj: dict[str, Any]) -> str:
    name = (tool_name or "").strip()
    if name == "write":
        return fingerprint_write(
            str(input_obj.get("path") or ""),
            str(input_obj.get("content") or ""),
        )
    if name == "edit":
        return fingerprint_edit(
            str(input_obj.get("path") or ""),
            input_obj.get("edits") or [],
        )
    if name in {"bash", "powershell"}:
        return fingerprint_shell(name, str(input_obj.get("command") or ""))
    return f"{name}:{_sha(json.dumps(input_obj, ensure_ascii=False, sort_keys=True))}"


_FP_LINE = re.compile(r"(?m)^fp=(\S+)\s*$")
_WRITE_HEAD = re.compile(
    r"(?m)^path=(?P<path>\S+)\s+content_len=(?P<clen>\d+)\s*$"
)
_EDIT_HEAD = re.compile(r"(?m)^path=(?P<path>\S+)\s+edits=(?P<n>\d+)\s*$")


def fingerprint_from_confirm_message(message: str | None) -> str | None:
    """Recover a stable id from gate confirm body (prefers embedded ``fp=``)."""
    text = message or ""
    m = _FP_LINE.search(text)
    if m:
        return m.group(1)
    # Fallback: hash the structured preview (path + length / edits count + body).
    wm = _WRITE_HEAD.search(text)
    if wm:
        # Include a short content head after --- so identical len≠identical body.
        head = ""
        if "\n---\n" in text:
            head = text.split("\n---\n", 1)[1][:400]
        return fingerprint_write(wm.group("path"), f"{wm.group('clen')}:{head}")
    em = _EDIT_HEAD.search(text)
    if em:
        return fingerprint_edit(em.group("path"), text.split("\n", 1)[-1][:800])
    if text.strip():
        return f"msg:{_sha(text.strip())}"
    return None


@dataclass
class RewriteGuard:
    """Session policy for duplicate gated confirms.

    - Already-approved fingerprint → auto-reject and abort (rewrite after success).
    - Same pending fingerprint twice in a row → auto-reject; third → abort.
    """

    approved: set[str] = field(default_factory=set)
    last_pending: str | None = None
    pending_streak: int = 0
    identical_pending_limit: int = 2
    identical_pending_abort_at: int = 3

    def on_pending(self, fp: str | None) -> tuple[Action, str]:
        if not fp:
            return "wait", ""
        if fp in self.approved:
            return (
                "abort",
                "duplicate of already-approved mutation; stopping rewrite loop",
            )
        if fp == self.last_pending:
            self.pending_streak += 1
        else:
            self.last_pending = fp
            self.pending_streak = 1
        if self.pending_streak >= self.identical_pending_abort_at:
            return (
                "abort",
                f"identical gated request repeated {self.pending_streak}×; abort",
            )
        if self.pending_streak >= self.identical_pending_limit:
            return (
                "auto_reject",
                f"identical gated request repeated {self.pending_streak}×; auto-reject",
            )
        return "wait", ""

    def on_approved(self, fp: str | None) -> None:
        if fp:
            self.approved.add(fp)
        self.pending_streak = 0
        self.last_pending = None

    def on_rejected(self, fp: str | None) -> None:
        # Keep streak / last_pending so a follow-up identical reject can escalate.
        del fp
