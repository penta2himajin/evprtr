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


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
            elif isinstance(block, str) and block.strip():
                parts.append(block.strip())
        return "\n".join(parts)
    return ""


def original_user_text(request: dict[str, Any]) -> str:
    """Prefer the primary user task, not later short steer/meta user turns.

    Multi-turn agent requests often append brief user messages after tool
    results. Joining everything diluted the task into noise; taking the
    longest early user message keeps repair grounded.
    """
    candidates: list[tuple[int, int, str]] = []
    for idx, message in enumerate(request.get("messages") or []):
        if not isinstance(message, dict):
            continue
        if message.get("role") != "user":
            continue
        text = _message_text(message.get("content"))
        if not text:
            continue
        # Skip tiny placeholders that confused Maple into "(no user text)" loops.
        if text.lower() in {"agent", "(no user text)", "no user text"}:
            continue
        candidates.append((idx, len(text), text))
    if not candidates:
        return "(no user text)"
    # Prefer earliest message among those near-max length (primary prompt).
    max_len = max(length for _, length, _ in candidates)
    primary = [c for c in candidates if c[1] >= max(80, int(max_len * 0.6))]
    pool = primary or candidates
    pool.sort(key=lambda item: (item[0], -item[1]))
    return pool[0][2]


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
        agentic = request_has_tools(request)

        if diagnosis.kind == "pseudo_tool_markup":
            if agentic:
                user = (
                    "Previous output illegally embedded a <tool_call> / XML-like block "
                    "inside assistant text instead of using the API tool channel.\n"
                    "Retry the coding-agent task.\n"
                    "If you need a tool, emit a real structured tool call only "
                    "(OpenAI tools). Do not invent markup in the text body.\n"
                    "Keep reasoning short.\n\n"
                    f"Task:\n{original}"
                )
                return self._apply_messages_and_knobs(
                    payload, user, attempt, agentic=True
                )
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

        if diagnosis.kind == "degenerate_tool_args":
            detail = diagnosis.detail or {}
            user = (
                "Previous structured tool call had unusable arguments "
                f"({detail.get('tool')}: {detail.get('reason')}; "
                f"path={detail.get('path')!r}, content_len={detail.get('content_len')}).\n"
                "Retry the coding-agent task.\n"
                "For file updates prefer the write tool with path + FULL file content "
                "(valid TOML/Rust source). Do not put the filename in content. "
                "Do not invent paths like Tower/MIT. Do not use no-op edits.\n"
                "Keep reasoning short.\n\n"
                f"Task:\n{original}"
            )
            return self._apply_messages_and_knobs(
                payload, user, attempt, agentic=True
            )

        if diagnosis.kind == "missing_mutation":
            user = (
                "Previous turn did not perform the required file/shell mutation "
                "(only read/prose, or no tool call).\n"
                "Retry now. Emit a real structured write/edit (or bash only if "
                "unavoidable) tool call. Do not stop after read-only exploration.\n"
                "Keep reasoning short.\n\n"
                f"Task:\n{original}"
            )
            return self._apply_messages_and_knobs(
                payload, user, attempt, agentic=True
            )

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
        if diagnosis.kind in {
            "pseudo_tool_markup",
            "degenerate_tool_args",
            "missing_mutation",
        }:
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
