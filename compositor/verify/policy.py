"""Verify policy protocols.

Generic detectors stay small and reusable. Model-family heuristics and repair
wording live in separate policy modules so they can be peeled off or generalized
later without rewriting the compositor loop.
"""

from __future__ import annotations

from typing import Any, Protocol

from compositor.verify.types import Diagnosis


class Detector(Protocol):
    """Inspect an assistant message; return a diagnosis or None."""

    policy_id: str

    def diagnose(
        self,
        message: dict[str, Any],
        *,
        request: dict[str, Any] | None = None,
    ) -> Diagnosis | None: ...


class RepairStrategy(Protocol):
    """Build a fresh upstream chat.completions payload for a repair attempt."""

    policy_id: str

    def build_payload(
        self,
        request: dict[str, Any],
        *,
        attempt: int,
        diagnosis: Diagnosis,
    ) -> dict[str, Any]: ...
