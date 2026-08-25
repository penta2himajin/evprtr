"""Shared types for verify/repair. Keep policies free of compositor I/O."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Diagnosis:
    """A verify hit. ``policy_id`` names the peelable unit that produced it."""

    policy_id: str
    field: str
    kind: str
    onset: int = 0
    detail: dict[str, Any] = field(default_factory=dict)

    def event_detail(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "field": self.field,
            "kind": self.kind,
            "onset": self.onset,
            "detail": self.detail,
        }
