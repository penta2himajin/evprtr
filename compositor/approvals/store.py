"""Pending side-effect tool calls awaiting human/supervisor approval."""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class ApprovalStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    APPLIED = "applied"


@dataclass
class PendingAction:
    id: str
    created_at: float
    tool_name: str
    arguments: str
    raw_tool_call: dict[str, Any]
    status: str = ApprovalStatus.PENDING.value
    trace_id: str | None = None
    reason: str | None = None
    decision_note: str | None = None
    decided_at: float | None = None
    apply_result: dict[str, Any] | None = None
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ApprovalStore:
    def __init__(self, directory: Path | str | None = None, *, memory_limit: int = 500) -> None:
        self.directory = Path(directory) if directory else None
        self.memory_limit = memory_limit
        self._lock = threading.Lock()
        self._items: dict[str, PendingAction] = {}
        self._order: list[str] = []
        if self.directory:
            self.directory.mkdir(parents=True, exist_ok=True)

    def enqueue(
        self,
        *,
        tool_name: str,
        arguments: str,
        raw_tool_call: dict[str, Any],
        trace_id: str | None = None,
        reason: str | None = None,
        tags: list[str] | None = None,
    ) -> PendingAction:
        action = PendingAction(
            id=f"appr-{uuid.uuid4().hex[:20]}",
            created_at=time.time(),
            tool_name=tool_name,
            arguments=arguments,
            raw_tool_call=raw_tool_call,
            trace_id=trace_id,
            reason=reason,
            tags=tags or [],
        )
        self._save(action)
        return action

    def get(self, action_id: str) -> PendingAction | None:
        with self._lock:
            found = self._items.get(action_id)
        if found:
            return found
        if self.directory:
            path = self.directory / f"{action_id}.json"
            if path.is_file():
                data = json.loads(path.read_text(encoding="utf-8"))
                action = _from_dict(data)
                with self._lock:
                    self._items[action.id] = action
                return action
        return None

    def list(
        self,
        *,
        status: str | None = None,
        limit: int = 50,
    ) -> list[PendingAction]:
        with self._lock:
            ids = list(reversed(self._order))
            items = [self._items[i] for i in ids if i in self._items]
        if status:
            items = [a for a in items if a.status == status]
        return items[: max(1, min(limit, 200))]

    def decide(
        self,
        action_id: str,
        *,
        approve: bool,
        note: str | None = None,
    ) -> PendingAction | None:
        action = self.get(action_id)
        if action is None:
            return None
        if action.status not in {ApprovalStatus.PENDING.value, ApprovalStatus.APPROVED.value}:
            return action
        action.status = ApprovalStatus.APPROVED.value if approve else ApprovalStatus.REJECTED.value
        action.decision_note = note
        action.decided_at = time.time()
        self._save(action)
        return action

    def mark_applied(self, action_id: str, result: dict[str, Any]) -> PendingAction | None:
        action = self.get(action_id)
        if action is None:
            return None
        action.status = ApprovalStatus.APPLIED.value
        action.apply_result = result
        action.decided_at = action.decided_at or time.time()
        self._save(action)
        return action

    def _save(self, action: PendingAction) -> None:
        with self._lock:
            if action.id not in self._items:
                self._order.append(action.id)
            self._items[action.id] = action
            while len(self._order) > self.memory_limit:
                old = self._order.pop(0)
                self._items.pop(old, None)
        if self.directory:
            path = self.directory / f"{action.id}.json"
            path.write_text(
                json.dumps(action.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )


def _from_dict(data: dict[str, Any]) -> PendingAction:
    return PendingAction(
        id=data["id"],
        created_at=data["created_at"],
        tool_name=data["tool_name"],
        arguments=data.get("arguments") or "",
        raw_tool_call=data.get("raw_tool_call") or {},
        status=data.get("status", ApprovalStatus.PENDING.value),
        trace_id=data.get("trace_id"),
        reason=data.get("reason"),
        decision_note=data.get("decision_note"),
        decided_at=data.get("decided_at"),
        apply_result=data.get("apply_result"),
        tags=list(data.get("tags") or []),
    )
