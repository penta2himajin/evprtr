"""Generic detectors — safe defaults for any upstream generative runtime."""

from __future__ import annotations

from typing import Any

from compositor.verify.repetition import find_repetition, message_fields
from compositor.verify.types import Diagnosis

POLICY_ID = "generic.repetition"


class RepetitionDetector:
    """Flag word-run / n-gram / motif collapse in content or reasoning."""

    policy_id = POLICY_ID

    def diagnose(
        self,
        message: dict[str, Any],
        *,
        request: dict[str, Any] | None = None,
    ) -> Diagnosis | None:
        del request
        for field, text in message_fields(message).items():
            hit = find_repetition(text)
            if hit is not None:
                return Diagnosis(
                    policy_id=self.policy_id,
                    field=field,
                    kind=hit.kind,
                    onset=hit.onset,
                    detail=dict(hit.detail),
                )
        return None
