"""Verify/repair pipeline — orchestration only; policies are injected."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from compositor.verify.detectors_generic import RepetitionDetector
from compositor.verify.detectors_maple_preview import (
    EmptyContentLongReasoningDetector,
    PseudoToolCallInContentDetector,
    ThinContentDetector,
    strip_pseudo_tool_markup,
)
from compositor.verify.policy import Detector, RepairStrategy
from compositor.verify.repair import FreshConstrainedRepair, MaplePreviewRepair
from compositor.verify.repetition import find_repetition, truncate_before_repetition
from compositor.verify.types import Diagnosis


def assistant_message(upstream: dict[str, Any]) -> dict[str, Any] | None:
    choices = upstream.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return None
    message = choices[0].get("message")
    return message if isinstance(message, dict) else None


@dataclass
class VerifyBundle:
    """Injectable verify stack. Swap or drop policies without touching the loop."""

    detectors: list[Detector] = field(default_factory=list)
    repair: RepairStrategy = field(default_factory=FreshConstrainedRepair)
    repair_attempts: int = 2

    @staticmethod
    def generic_only(*, repair_attempts: int = 2) -> VerifyBundle:
        """Repetition detector + neutral repair. No Maple-specific heuristics."""
        return VerifyBundle(
            detectors=[RepetitionDetector()],
            repair=FreshConstrainedRepair(),
            repair_attempts=repair_attempts,
        )

    @staticmethod
    def maple_preview(*, repair_attempts: int = 2) -> VerifyBundle:
        """Default for current work: generic core + peelable Maple heuristics."""
        return VerifyBundle(
            detectors=[
                RepetitionDetector(),
                PseudoToolCallInContentDetector(),
                ThinContentDetector(),
                EmptyContentLongReasoningDetector(),
            ],
            repair=MaplePreviewRepair(),
            repair_attempts=repair_attempts,
        )

    def diagnose(
        self,
        upstream: dict[str, Any],
        *,
        request: dict[str, Any] | None = None,
    ) -> Diagnosis | None:
        message = assistant_message(upstream)
        if message is None:
            return None
        for detector in self.detectors:
            hit = detector.diagnose(message, request=request)
            if hit is not None:
                return hit
        return None

    def sanitize(self, upstream: dict[str, Any], diagnosis: Diagnosis) -> dict[str, Any]:
        out = copy.deepcopy(upstream)
        message = assistant_message(out)
        if message is None:
            return out

        if diagnosis.kind == "pseudo_tool_markup":
            raw = message.get("content")
            if isinstance(raw, str):
                message["content"] = strip_pseudo_tool_markup(raw)
        elif diagnosis.kind not in {"thin_content", "empty_content_long_reasoning"}:
            raw = message.get(diagnosis.field)
            if isinstance(raw, str):
                from compositor.verify.repetition import RepetitionHit

                hit = RepetitionHit(
                    kind=diagnosis.kind,
                    onset=diagnosis.onset,
                    detail={k: v for k, v in diagnosis.detail.items() if isinstance(v, (str, int))},
                )
                message[diagnosis.field] = truncate_before_repetition(raw, hit)

        reasoning = message.get("reasoning_content")
        if isinstance(reasoning, str):
            rhit = find_repetition(reasoning)
            if rhit is not None:
                truncated = truncate_before_repetition(reasoning, rhit)
                message["reasoning_content"] = truncated or None
        if not isinstance(message.get("content"), str) or not str(message.get("content")).strip():
            message["content"] = message.get("content") or ""
        return out

    def has_usable_content(
        self,
        upstream: dict[str, Any],
        *,
        request: dict[str, Any] | None = None,
    ) -> bool:
        message = assistant_message(upstream)
        if message is None:
            return False
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            return False
        if find_repetition(content) is not None:
            return False
        for detector in self.detectors:
            if detector.policy_id.startswith("generic."):
                continue
            if detector.diagnose(message, request=request) is not None:
                return False
        return True

    def repair_payload(
        self, request: dict[str, Any], *, attempt: int, diagnosis: Diagnosis
    ) -> dict[str, Any]:
        return self.repair.build_payload(request, attempt=attempt, diagnosis=diagnosis)
