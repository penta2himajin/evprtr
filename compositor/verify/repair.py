"""Repair strategies.

``FreshConstrainedRepair`` is the generic skeleton (new short turn + sampling
knobs). ``MaplePreviewRepair`` only adds Maple-oriented prompt wording and can
be swapped for a neutral strategy later.

When the original request includes tools, repair must stay agent-shaped:
prefer a real tool call over a prose-only “answer body”.
"""

from __future__ import annotations

import copy
from typing import Any

from compositor.verify.types import Diagnosis

GENERIC_REPAIR_ID = "generic.fresh_constrained"
MAPLE_REPAIR_ID = "maple_preview.fresh_constrained"


def original_user_text(request: dict[str, Any]) -> str:
    parts: list[str] = []
    for message in request.get("messages") or []:
        if not isinstance(message, dict):
            continue
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            parts.append(content.strip())
    return "\n\n".join(parts) if parts else "(no user text)"


def request_has_tools(request: dict[str, Any]) -> bool:
    tools = request.get("tools")
    if not isinstance(tools, list) or not tools:
        return False
    choice = request.get("tool_choice")
    if choice in {"none", False}:
        return False
    return True


class FreshConstrainedRepair:
    """Generic repair: discard broken context, tighten sampling, short answer."""

    policy_id = GENERIC_REPAIR_ID

    def build_payload(
        self,
        request: dict[str, Any],
        *,
        attempt: int,
        diagnosis: Diagnosis,
    ) -> dict[str, Any]:
        payload = copy.deepcopy(request)
        original = original_user_text(request)
        if diagnosis.kind == "pseudo_tool_markup":
            user = (
                "Previous output illegally embedded a <tool_call> block inside assistant text.\n"
                "Answer again in natural language only.\n"
                "Do not emit tool markup or JSON tool calls inside the text body.\n"
                "If tools are disallowed or unavailable, say so briefly and answer "
                "without live tool data.\n\n"
                f"Task:\n{original}"
            )
            payload = self._apply_messages_and_knobs(
                payload, user, attempt, agentic=False
            )
            payload["tool_choice"] = "none"
            return payload

        agentic = request_has_tools(request)
        if agentic:
            user = (
                "Previous model output was unusable (repetition, empty answer, or confusion).\n"
                "Retry the coding-agent task from scratch.\n"
                "Prefer a real structured tool call (OpenAI tools) when the task needs "
                "filesystem or shell work.\n"
                "Do not invent XML/pseudo <tool_call> markup in text.\n"
                "Keep reasoning short; avoid runaway repetition.\n\n"
                f"Task:\n{original}"
            )
        elif attempt <= 1:
            user = (
                "Previous model output was unusable (repetition or empty answer).\n"
                "Do the task again from scratch.\n"
                "Rules: final answer only; no long analysis; "
                "avoid runaway token-repetition loops.\n\n"
                f"Task:\n{original}"
            )
        else:
            user = (
                "Retry the task. Keep the answer short and concrete. "
                "Avoid runaway repetition.\n\n"
                f"Task:\n{original}"
            )
        return self._apply_messages_and_knobs(
            payload, user, attempt, agentic=agentic
        )

    def _apply_messages_and_knobs(
        self,
        payload: dict[str, Any],
        user: str,
        attempt: int,
        *,
        agentic: bool,
    ) -> dict[str, Any]:
        if agentic:
            system = (
                "You are a coding agent. Use the provided tools via structured "
                "tool calls when the task requires edits or commands. "
                "Avoid runaway token repetition. Do not pad with filler."
            )
        else:
            system = (
                "You write concise correct answers. "
                "Avoid runaway token repetition. Do not pad with filler."
            )
        payload["messages"] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        payload["temperature"] = 0.2 if attempt <= 1 else 0.1
        payload["frequency_penalty"] = 0.8 if attempt <= 1 else 1.0
        payload["presence_penalty"] = 0.6 if attempt <= 1 else 0.8
        payload["repeat_penalty"] = 1.3 if attempt <= 1 else 1.45
        if agentic:
            # Need room for at least one tool call + short rationale.
            payload["max_tokens"] = 1024 if attempt <= 1 else 768
        else:
            payload["max_tokens"] = 384 if attempt <= 1 else 256
        payload["stream"] = False
        return payload


class MaplePreviewRepair(FreshConstrainedRepair):
    """Maple-oriented prompt wording on top of the generic constrained repair."""

    policy_id = MAPLE_REPAIR_ID

    def build_payload(
        self,
        request: dict[str, Any],
        *,
        attempt: int,
        diagnosis: Diagnosis,
    ) -> dict[str, Any]:
        if diagnosis.kind == "pseudo_tool_markup":
            return super().build_payload(request, attempt=attempt, diagnosis=diagnosis)

        if request_has_tools(request):
            # Same agentic path as generic — avoid “Emit ONLY answer body” which
            # makes Maple ignore tools and reply with Ready / empty content.
            return FreshConstrainedRepair.build_payload(
                self, request, attempt=attempt, diagnosis=diagnosis
            )

        payload = copy.deepcopy(request)
        original = original_user_text(request)
        if attempt <= 1:
            user = (
                "Previous model output collapsed into repeated tokens or was unusable.\n"
                "Do the task again from scratch.\n"
                "Rules: final answer only; no long analysis; "
                "avoid runaway token-repetition loops;\n"
                "if bullets were asked, emit the bullets now with no intro.\n\n"
                f"Task:\n{original}"
            )
        else:
            user = (
                "Retry with a short concrete answer. Avoid runaway repetition.\n"
                "Prefer short bullet lines when listing.\n\n"
                f"Task:\n{original}"
            )
        return self._apply_messages_and_knobs(
            payload, user, attempt, agentic=False
        )
