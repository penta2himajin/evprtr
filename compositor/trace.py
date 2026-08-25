"""Failure / request traces for debug and improvement feedback.

Traces record *where* something failed (locus), not leaderboard scores.
The compositor calls runtimes; traces observe that path.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class FailureLocus(StrEnum):
    """Stage that failed. Extend as composition grows."""

    NONE = "none"
    ACCEPT = "accept"
    UPSTREAM = "upstream"
    PRESENT = "present"
    VERIFY = "verify"
    REPAIR = "repair"
    # Reserved for later composition stages:
    # PLAN = "plan"
    # TOOL_SELECT = "tool_select"
    # EXECUTE = "execute"


@dataclass
class TraceEvent:
    stage: str
    status: str  # "ok" | "error"
    at: float
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class TraceRecord:
    trace_id: str
    started_at: float
    finished_at: float | None = None
    public_model: str = "evprtr"
    ok: bool = True
    locus: str = FailureLocus.NONE.value
    error: str | None = None
    request_summary: dict[str, Any] = field(default_factory=dict)
    events: list[TraceEvent] = field(default_factory=list)
    response_summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize_request(request: dict[str, Any]) -> dict[str, Any]:
    messages = request.get("messages") or []
    return {
        "message_count": len(messages),
        "roles": [m.get("role") for m in messages if isinstance(m, dict)],
        "has_tools": bool(request.get("tools")),
        "tool_choice": request.get("tool_choice"),
        "stream": bool(request.get("stream")),
        "requested_model": request.get("model"),
    }


def summarize_response(response: dict[str, Any]) -> dict[str, Any]:
    choices = response.get("choices") or []
    finish_reasons = []
    has_tool_calls = False
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        finish_reasons.append(choice.get("finish_reason"))
        message = choice.get("message") or {}
        if isinstance(message, dict) and message.get("tool_calls"):
            has_tool_calls = True
    usage = response.get("usage") or {}
    return {
        "choice_count": len(choices),
        "finish_reasons": finish_reasons,
        "has_tool_calls": has_tool_calls,
        "usage": usage if isinstance(usage, dict) else {},
    }


class TraceSession:
    def __init__(
        self,
        store: TraceStore,
        *,
        public_model: str,
        request_summary: dict[str, Any] | None = None,
        trace_id: str | None = None,
    ) -> None:
        self._store = store
        self.record = TraceRecord(
            trace_id=trace_id or f"tr-{uuid.uuid4().hex}",
            started_at=time.time(),
            public_model=public_model,
            request_summary=request_summary or {},
        )

    @property
    def trace_id(self) -> str:
        return self.record.trace_id

    def event(self, stage: str, status: str = "ok", **detail: Any) -> None:
        self.record.events.append(
            TraceEvent(stage=stage, status=status, at=time.time(), detail=detail)
        )

    def fail(self, locus: FailureLocus | str, error: str, *, stage: str | None = None) -> None:
        locus_value = locus.value if isinstance(locus, FailureLocus) else locus
        self.record.ok = False
        self.record.locus = locus_value
        self.record.error = error
        self.event(stage or locus_value, status="error", error=error, locus=locus_value)

    def finish(
        self,
        *,
        ok: bool | None = None,
        response_summary: dict[str, Any] | None = None,
    ) -> TraceRecord:
        if ok is not None:
            self.record.ok = ok
            if ok:
                self.record.locus = FailureLocus.NONE.value
        if response_summary is not None:
            self.record.response_summary = response_summary
        self.record.finished_at = time.time()
        self._store.save(self.record)
        return self.record


class TraceStore:
    def __init__(
        self,
        directory: Path | str | None = None,
        *,
        memory_limit: int = 200,
    ) -> None:
        self.directory = Path(directory) if directory else None
        self.memory_limit = memory_limit
        self._lock = threading.Lock()
        self._records: dict[str, TraceRecord] = {}
        self._order: list[str] = []
        if self.directory:
            self.directory.mkdir(parents=True, exist_ok=True)

    def start(
        self,
        *,
        public_model: str,
        request_summary: dict[str, Any] | None = None,
        trace_id: str | None = None,
    ) -> TraceSession:
        return TraceSession(
            self,
            public_model=public_model,
            request_summary=request_summary,
            trace_id=trace_id,
        )

    def save(self, record: TraceRecord) -> None:
        with self._lock:
            if record.trace_id not in self._records:
                self._order.append(record.trace_id)
            self._records[record.trace_id] = record
            while len(self._order) > self.memory_limit:
                old = self._order.pop(0)
                self._records.pop(old, None)
        if self.directory:
            path = self.directory / f"{record.trace_id}.json"
            path.write_text(
                json.dumps(record.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    def get(self, trace_id: str) -> TraceRecord | None:
        with self._lock:
            found = self._records.get(trace_id)
        if found:
            return found
        if self.directory:
            path = self.directory / f"{trace_id}.json"
            if path.is_file():
                data = json.loads(path.read_text(encoding="utf-8"))
                return _record_from_dict(data)
        return None

    def list_recent(self, limit: int = 20) -> list[TraceRecord]:
        with self._lock:
            ids = list(reversed(self._order[-limit:]))
            return [self._records[i] for i in ids if i in self._records]


def _record_from_dict(data: dict[str, Any]) -> TraceRecord:
    events = [
        TraceEvent(
            stage=e["stage"],
            status=e["status"],
            at=e["at"],
            detail=e.get("detail") or {},
        )
        for e in data.get("events") or []
    ]
    return TraceRecord(
        trace_id=data["trace_id"],
        started_at=data["started_at"],
        finished_at=data.get("finished_at"),
        public_model=data.get("public_model", "evprtr"),
        ok=bool(data.get("ok", True)),
        locus=data.get("locus", FailureLocus.NONE.value),
        error=data.get("error"),
        request_summary=data.get("request_summary") or {},
        events=events,
        response_summary=data.get("response_summary") or {},
    )
