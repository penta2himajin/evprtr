"""Buffer side-effect tool_calls into an approval queue before harness execution."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any

from compositor.approvals.classify import split_tool_calls, tool_name
from compositor.approvals.store import ApprovalStore, PendingAction


@dataclass
class BufferOutcome:
    response: dict[str, Any]
    buffered: list[PendingAction]
    passed_through: list[dict[str, Any]]


class SideEffectBuffer:
    """Peelable policy: intercept risky tool_calls, enqueue for review."""

    policy_id = "approvals.side_effect_buffer"

    def __init__(self, store: ApprovalStore, *, enabled: bool = True) -> None:
        self.store = store
        self.enabled = enabled

    def process(
        self,
        response: dict[str, Any],
        *,
        trace_id: str | None = None,
    ) -> BufferOutcome:
        if not self.enabled:
            return BufferOutcome(response=response, buffered=[], passed_through=[])

        out = copy.deepcopy(response)
        choices = out.get("choices") or []
        if not choices or not isinstance(choices[0], dict):
            return BufferOutcome(response=out, buffered=[], passed_through=[])

        message = choices[0].get("message")
        if not isinstance(message, dict):
            return BufferOutcome(response=out, buffered=[], passed_through=[])

        tool_calls = message.get("tool_calls")
        if not tool_calls:
            return BufferOutcome(response=out, buffered=[], passed_through=[])

        safe, risky = split_tool_calls(tool_calls)
        buffered: list[PendingAction] = []
        for call in risky:
            name = tool_name(call)
            fn = call.get("function") if isinstance(call.get("function"), dict) else {}
            args = fn.get("arguments") if isinstance(fn, dict) else call.get("arguments")
            if not isinstance(args, str):
                args = json.dumps(args or {}, ensure_ascii=False)
            action = self.store.enqueue(
                tool_name=name,
                arguments=args,
                raw_tool_call=call,
                trace_id=trace_id,
                reason="side-effect tool buffered for review before apply",
                tags=["side_effect"],
            )
            buffered.append(action)

        if not buffered:
            return BufferOutcome(response=out, buffered=[], passed_through=safe)

        # Do not hand side-effect calls to the harness; keep only safe tools.
        if safe:
            message["tool_calls"] = safe
            choices[0]["finish_reason"] = "tool_calls"
        else:
            message["tool_calls"] = None
            choices[0]["finish_reason"] = "stop"
            summary = _pending_summary(buffered)
            existing = message.get("content")
            if isinstance(existing, str) and existing.strip():
                message["content"] = existing.rstrip() + "\n\n" + summary
            else:
                message["content"] = summary

        return BufferOutcome(response=out, buffered=buffered, passed_through=safe)


def _pending_summary(actions: list[PendingAction]) -> str:
    lines = [
        "Side-effect tool calls were buffered for approval (not executed):",
    ]
    for action in actions:
        preview = action.arguments.replace("\n", " ")
        if len(preview) > 160:
            preview = preview[:157] + "..."
        lines.append(f"- [{action.id}] {action.tool_name}: {preview}")
    lines.append("Review via GET /v1/approvals and POST /v1/approvals/{id}/approve|reject.")
    return "\n".join(lines)
