"""Maple-Preview–oriented heuristics.

These matched failures seen in early Pi/Maple probes. They are intentionally
isolated so they can be disabled, replaced, or generalized to other small
reasoning models without touching the verify loop.

See docs/findings/maple-preview-probe-2026-08-25.md for live probe evidence.
"""

from __future__ import annotations

import re
from typing import Any

from compositor.verify.repetition import find_repetition
from compositor.verify.types import Diagnosis

THIN_CONTENT_ID = "maple_preview.thin_content"
EMPTY_LONG_REASONING_ID = "maple_preview.empty_content_long_reasoning"
PSEUDO_TOOL_MARKUP_ID = "maple_preview.pseudo_tool_markup"

_PSEUDO_TOOL_BLOCK = re.compile(
    r"<tool_call\b[^>]*>[\s\S]*?</tool_call>|</?tool_call\b[^>]*>",
    re.IGNORECASE,
)


class ThinContentDetector:
    """Preamble-only answers (e.g. ends with ':' and never delivers bullets)."""

    policy_id = THIN_CONTENT_ID

    def diagnose(
        self,
        message: dict[str, Any],
        *,
        request: dict[str, Any] | None = None,
    ) -> Diagnosis | None:
        del request
        content = message.get("content")
        if not isinstance(content, str):
            return None
        text = content.strip()
        if not text:
            return None
        if text.endswith(":") and not any(
            marker in text for marker in ("\n-", "\n*", "\n1.", "\n1)")
        ):
            return Diagnosis(
                policy_id=self.policy_id,
                field="content",
                kind="thin_content",
                onset=0,
                detail={"len": len(text)},
            )
        return None


class EmptyContentLongReasoningDetector:
    """Empty content with a long reasoning blob (common Maple collapse shape)."""

    policy_id = EMPTY_LONG_REASONING_ID

    def __init__(self, *, min_reasoning_len: int = 800) -> None:
        self.min_reasoning_len = min_reasoning_len

    def diagnose(
        self,
        message: dict[str, Any],
        *,
        request: dict[str, Any] | None = None,
    ) -> Diagnosis | None:
        del request
        content = message.get("content")
        reasoning = message.get("reasoning_content")
        content_s = content.strip() if isinstance(content, str) else ""
        if content_s:
            return None
        if not isinstance(reasoning, str) or len(reasoning) <= self.min_reasoning_len:
            return None
        hit = find_repetition(reasoning)
        if hit is not None:
            return Diagnosis(
                policy_id=self.policy_id,
                field="reasoning_content",
                kind=hit.kind,
                onset=hit.onset,
                detail={**dict(hit.detail), "reasoning_len": len(reasoning)},
            )
        return Diagnosis(
            policy_id=self.policy_id,
            field="reasoning_content",
            kind="empty_content_long_reasoning",
            onset=min(400, len(reasoning)),
            detail={"reasoning_len": len(reasoning)},
        )


class PseudoToolCallInContentDetector:
    """Tool markup leaked into assistant content instead of structured tool_calls.

    Observed when tool_choice=none: model still writes <tool_call>...</tool_call>
    into content. Harness-breaking and peelable for other small agentic models.
    """

    policy_id = PSEUDO_TOOL_MARKUP_ID

    def diagnose(
        self,
        message: dict[str, Any],
        *,
        request: dict[str, Any] | None = None,
    ) -> Diagnosis | None:
        del request
        content = message.get("content")
        if not isinstance(content, str) or "<tool_call" not in content.lower():
            return None
        # Structured tool_calls present does not excuse markup pollution in text.
        match = _PSEUDO_TOOL_BLOCK.search(content)
        onset = match.start() if match else content.lower().index("<tool_call")
        return Diagnosis(
            policy_id=self.policy_id,
            field="content",
            kind="pseudo_tool_markup",
            onset=onset,
            detail={
                "has_structured_tool_calls": bool(message.get("tool_calls")),
            },
        )


def strip_pseudo_tool_markup(text: str) -> str:
    cleaned = _PSEUDO_TOOL_BLOCK.sub("", text)
    return cleaned.strip()
