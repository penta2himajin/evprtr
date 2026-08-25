"""Composite layer orchestration.

Calls local model runtimes (Maple for prose, Needle for tool determination when
enabled), runs an injectable verify/repair stack, presents one OpenAI-compatible
face. Distillation is out of scope.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass
from typing import Any

from compositor.approvals.buffer import SideEffectBuffer
from compositor.approvals.store import ApprovalStore
from compositor.runtimes import ChatRuntime, UpstreamError
from compositor.tools.path import NeedleToolPath
from compositor.trace import (
    FailureLocus,
    TraceRecord,
    TraceSession,
    TraceStore,
    summarize_request,
    summarize_response,
)
from compositor.verify.pipeline import VerifyBundle


@dataclass
class ComposeResult:
    response: dict[str, Any]
    trace: TraceRecord


class Compositor:
    def __init__(
        self,
        runtime: ChatRuntime,
        *,
        public_model_id: str = "evprtr",
        traces: TraceStore | None = None,
        verify: VerifyBundle | None = None,
        repair_attempts: int | None = None,
        tool_path: NeedleToolPath | None = None,
        approvals: ApprovalStore | None = None,
        side_effect_buffer: SideEffectBuffer | None = None,
        buffer_side_effects: bool = True,
    ) -> None:
        self.runtime = runtime
        self.public_model_id = public_model_id
        self.traces = traces or TraceStore()
        # Default stack targets Maple-Preview today; peel maple_* policies later.
        self.verify = verify or VerifyBundle.maple_preview()
        if repair_attempts is not None:
            self.verify.repair_attempts = max(0, repair_attempts)
        self.tool_path = tool_path
        self.approvals = approvals or ApprovalStore()
        self.side_effect_buffer = side_effect_buffer or SideEffectBuffer(
            self.approvals, enabled=buffer_side_effects
        )

    async def chat_completions(self, request: dict[str, Any]) -> ComposeResult:
        session = self.traces.start(
            public_model=self.public_model_id,
            request_summary=summarize_request(request),
        )
        session.event("accept", "ok")

        tool_result = await self._maybe_tool_path(request, session)
        if tool_result is not None:
            try:
                presented = self._present(tool_result.response)
                presented, buffered_ids = self._buffer_side_effects(
                    presented, session, session.trace_id
                )
                session.event("present", "ok", via="needle_tool_path")
            except Exception as exc:  # noqa: BLE001
                session.fail(FailureLocus.PRESENT, str(exc), stage="present")
                session.finish(ok=False)
                raise
            summary = summarize_response(presented)
            summary["repetition_repaired"] = False
            summary["needle_tool_path"] = True
            summary["needle_empty_call"] = tool_result.empty_call
            summary["needle_confidence"] = tool_result.confidence
            summary["buffered_approvals"] = buffered_ids
            trace = session.finish(ok=True, response_summary=summary)
            return ComposeResult(response=presented, trace=trace)

        try:
            session.event("upstream_call", "ok", phase="start", runtime="maple")
            upstream = await self.runtime.chat_completions(request)
            session.event("upstream_call", "ok", phase="done", runtime="maple")
        except UpstreamError as exc:
            session.fail(FailureLocus.UPSTREAM, str(exc), stage="upstream_call")
            session.finish(ok=False)
            raise
        except Exception as exc:  # noqa: BLE001 — last-resort locus tagging
            session.fail(FailureLocus.UPSTREAM, f"unexpected: {exc}", stage="upstream_call")
            session.finish(ok=False)
            raise

        upstream, repaired = await self._verify_and_repair(request, upstream, session)

        try:
            presented = self._present(upstream)
            presented, buffered_ids = self._buffer_side_effects(
                presented, session, session.trace_id
            )
            session.event("present", "ok", repaired=repaired)
        except Exception as exc:  # noqa: BLE001
            session.fail(FailureLocus.PRESENT, str(exc), stage="present")
            session.finish(ok=False)
            raise

        summary = summarize_response(presented)
        summary["repetition_repaired"] = repaired
        summary["needle_tool_path"] = False
        summary["buffered_approvals"] = buffered_ids
        trace = session.finish(ok=True, response_summary=summary)
        return ComposeResult(response=presented, trace=trace)

    def _buffer_side_effects(
        self,
        presented: dict[str, Any],
        session: TraceSession,
        trace_id: str,
    ) -> tuple[dict[str, Any], list[str]]:
        outcome = self.side_effect_buffer.process(presented, trace_id=trace_id)
        if outcome.buffered:
            session.event(
                "approval_buffer",
                "ok",
                policy_id=self.side_effect_buffer.policy_id,
                buffered=[a.id for a in outcome.buffered],
                tools=[a.tool_name for a in outcome.buffered],
            )
        return outcome.response, [a.id for a in outcome.buffered]

    async def _maybe_tool_path(self, request: dict[str, Any], session: TraceSession):
        if self.tool_path is None:
            return None
        if not self.tool_path.enabled_for(request):
            return None
        session.event(
            "tool_select",
            "ok",
            phase="start",
            policy_id=self.tool_path.policy_id,
        )
        try:
            result = await asyncio.to_thread(self.tool_path.handle, request)
        except Exception as exc:  # noqa: BLE001
            session.event(
                "tool_select",
                "error",
                policy_id=self.tool_path.policy_id,
                error=str(exc),
            )
            # Fall back to Maple rather than failing the whole request.
            return None
        if result is None:
            session.event(
                "tool_select",
                "ok",
                phase="fallback_maple",
                policy_id=self.tool_path.policy_id,
            )
            return None
        session.event(
            "tool_select",
            "ok",
            phase="done",
            policy_id=self.tool_path.policy_id,
            empty_call=result.empty_call,
            confidence=result.confidence,
            call_count=len(
                ((result.response.get("choices") or [{}])[0].get("message") or {}).get("tool_calls")
                or []
            ),
        )
        return result

    async def _verify_and_repair(
        self,
        request: dict[str, Any],
        upstream: dict[str, Any],
        session: TraceSession,
    ) -> tuple[dict[str, Any], bool]:
        current = upstream
        repaired = False
        attempts = self.verify.repair_attempts

        for attempt in range(attempts + 1):
            diagnosis = self.verify.diagnose(current, request=request)
            if diagnosis is None:
                session.event("verify", "ok", attempt=attempt)
                return current, repaired

            session.event("verify", "error", attempt=attempt, **diagnosis.event_detail())

            sanitized = self.verify.sanitize(current, diagnosis)
            # Pseudo tool markup must be repaired, not merely stripped — leftover
            # "I'll get the weather..." prose still misleads harnesses.
            if diagnosis.kind != "pseudo_tool_markup" and self.verify.has_usable_content(
                sanitized, request=request
            ):
                session.event(
                    "verify",
                    "ok",
                    attempt=attempt,
                    action="truncate",
                    policy_id=diagnosis.policy_id,
                )
                return sanitized, True

            if attempt >= attempts:
                session.event(
                    "verify",
                    "error",
                    attempt=attempt,
                    action="exhausted",
                    **diagnosis.event_detail(),
                )
                return sanitized, True

            session.event(
                "repair",
                "ok",
                phase="start",
                attempt=attempt + 1,
                policy_id=self.verify.repair.policy_id,
            )
            try:
                current = await self.runtime.chat_completions(
                    self.verify.repair_payload(request, attempt=attempt + 1, diagnosis=diagnosis)
                )
                repaired = True
                session.event(
                    "repair",
                    "ok",
                    phase="done",
                    attempt=attempt + 1,
                    policy_id=self.verify.repair.policy_id,
                )
            except UpstreamError as exc:
                session.fail(FailureLocus.REPAIR, str(exc), stage="repair")
                session.finish(ok=False)
                raise

        return current, repaired

    def _present(self, upstream: dict[str, Any]) -> dict[str, Any]:
        presented = dict(upstream)
        presented["id"] = presented.get("id") or f"chatcmpl-{uuid.uuid4().hex[:24]}"
        presented["object"] = "chat.completion"
        presented["created"] = int(presented.get("created") or time.time())
        presented["model"] = self.public_model_id
        if "choices" not in presented:
            presented["choices"] = []
        return presented
