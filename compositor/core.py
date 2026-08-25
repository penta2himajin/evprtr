"""Composite layer orchestration.

Calls local model runtimes (Maple for prose / NL plans, Needle for structured
tool calls when enabled), runs an injectable verify/repair stack, presents one
OpenAI-compatible face. Distillation is out of scope.
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from dataclasses import dataclass
from functools import partial
from typing import Any

from compositor.approvals.buffer import SideEffectBuffer
from compositor.approvals.store import ApprovalStore
from compositor.runtimes import ChatRuntime, UpstreamError
from compositor.tools.maple_nl import (
    assistant_text,
    correction_instruction,
    extract_fenced_files,
    maple_nl_request,
)
from compositor.tools.path import NeedleToolPath, ToolPathResult
from compositor.trace import (
    FailureLocus,
    TraceRecord,
    TraceSession,
    TraceStore,
    summarize_request,
    summarize_response,
)
from compositor.verify.pipeline import VerifyBundle
from compositor.verify.repair import original_user_text


def _env_flag(name: str, default: str = "1") -> bool:
    return (os.environ.get(name, default) or default).lower() not in {
        "0",
        "false",
        "off",
        "no",
    }


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
        needle_via_maple_nl: bool | None = None,
        needle_correct_degenerate: bool | None = None,
        needle_chunk_writes: bool | None = None,
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
        self.needle_via_maple_nl = (
            _env_flag("EVPRTR_NEEDLE_VIA_MAPLE_NL", "1")
            if needle_via_maple_nl is None
            else needle_via_maple_nl
        )
        self.needle_correct_degenerate = (
            _env_flag("EVPRTR_NEEDLE_CORRECT_DEGENERATE", "1")
            if needle_correct_degenerate is None
            else needle_correct_degenerate
        )
        self.needle_chunk_writes = (
            _env_flag("EVPRTR_NEEDLE_CHUNK_WRITES", "1")
            if needle_chunk_writes is None
            else needle_chunk_writes
        )

    async def chat_completions(self, request: dict[str, Any]) -> ComposeResult:
        session = self.traces.start(
            public_model=self.public_model_id,
            request_summary=summarize_request(request),
        )
        session.event("accept", "ok")

        # Preferred: Maple NL plan → Needle structures (and optional chunk apply).
        nl_result = await self._maybe_maple_nl_needle(request, session)
        if nl_result is not None:
            return await self._finish_tool_result(nl_result, session, request)

        tool_result = await self._maybe_tool_path(request, session)
        if tool_result is not None:
            return await self._finish_tool_result(tool_result, session, request)

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

    async def _finish_tool_result(
        self,
        tool_result: ToolPathResult,
        session: TraceSession,
        request: dict[str, Any],
    ) -> ComposeResult:
        current = tool_result.response
        try:
            diagnosis = self.verify.diagnose(current, request=request)
            if diagnosis is not None and diagnosis.kind == "degenerate_tool_args":
                session.event("verify", "error", **diagnosis.event_detail())
                sanitized = self.verify.sanitize(current, diagnosis)
                corrected = await self._needle_correct_degenerate(
                    request, diagnosis, sanitized, session, attempt=0
                )
                current = corrected if corrected is not None else sanitized
            presented = self._present(current)
            presented, buffered_ids = self._buffer_side_effects(
                presented, session, session.trace_id
            )
            session.event(
                "present",
                "ok",
                via=getattr(tool_result, "via", "needle_tool_path"),
            )
        except Exception as exc:  # noqa: BLE001
            session.fail(FailureLocus.PRESENT, str(exc), stage="present")
            session.finish(ok=False)
            raise
        summary = summarize_response(presented)
        summary["repetition_repaired"] = False
        summary["needle_tool_path"] = bool(tool_result.used_needle) or True
        summary["needle_empty_call"] = tool_result.empty_call
        summary["needle_confidence"] = tool_result.confidence
        summary["needle_via"] = getattr(tool_result, "via", None)
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

    async def _maybe_maple_nl_needle(
        self, request: dict[str, Any], session: TraceSession
    ) -> ToolPathResult | None:
        if not self.needle_via_maple_nl or self.tool_path is None:
            return None
        if not self.tool_path.enabled_for(request):
            return None

        session.event(
            "tool_select",
            "ok",
            phase="maple_nl_start",
            policy_id=self.tool_path.policy_id,
        )
        try:
            maple_out = await self.runtime.chat_completions(maple_nl_request(request))
        except UpstreamError:
            raise
        except Exception as exc:  # noqa: BLE001
            session.event(
                "tool_select",
                "error",
                phase="maple_nl",
                error=str(exc),
                policy_id=self.tool_path.policy_id,
            )
            return None

        instruction = assistant_text(maple_out)
        session.event(
            "tool_select",
            "ok",
            phase="maple_nl_done",
            instruction_len=len(instruction),
            policy_id=self.tool_path.policy_id,
        )
        if not instruction:
            session.event(
                "tool_select",
                "ok",
                phase="fallback_maple",
                reason="empty_maple_nl",
                policy_id=self.tool_path.policy_id,
            )
            return None

        if self.needle_chunk_writes:
            fenced = extract_fenced_files(instruction)
            for path, content in fenced:
                if len(content) < 12:
                    continue
                chunked = await asyncio.to_thread(
                    self.tool_path.apply_chunked_file, path, content, request
                )
                if chunked is not None:
                    session.event(
                        "tool_select",
                        "ok",
                        phase="done",
                        via=chunked.via,
                        chunks=True,
                        path=path,
                        policy_id=self.tool_path.policy_id,
                    )
                    return chunked

        result = await asyncio.to_thread(
            self.tool_path.handle_instruction, instruction, request
        )
        if result is None:
            session.event(
                "tool_select",
                "ok",
                phase="fallback_maple",
                reason=getattr(self.tool_path, "last_skip_reason", None),
                policy_id=self.tool_path.policy_id,
            )
            return None
        session.event(
            "tool_select",
            "ok",
            phase="done",
            via=result.via,
            empty_call=result.empty_call,
            confidence=result.confidence,
            policy_id=self.tool_path.policy_id,
        )
        return result

    async def _maybe_tool_path(self, request: dict[str, Any], session: TraceSession):
        if self.tool_path is None:
            return None
        if not self.tool_path.enabled_for(request):
            return None
        # When NL→Needle is enabled, direct user-text Needle is a second chance
        # only after Maple-with-tools; skip here to avoid double Needle on the
        # raw user prompt (often weaker than Maple NL).
        if self.needle_via_maple_nl:
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
            return None
        if result is None:
            session.event(
                "tool_select",
                "ok",
                phase="fallback_maple",
                policy_id=self.tool_path.policy_id,
                reason=getattr(self.tool_path, "last_skip_reason", None),
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
                ((result.response.get("choices") or [{}])[0].get("message") or {}).get(
                    "tool_calls"
                )
                or []
            ),
        )
        return result

    async def _verify_and_repair(
        self,
        request: dict[str, Any],
        upstream: dict[str, Any],
        session: TraceSession,
        *,
        allow_needle_correct: bool = True,
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
            # Pseudo markup / degenerate tool args must be repaired, not kept.
            if diagnosis.kind not in {
                "pseudo_tool_markup",
                "degenerate_tool_args",
            } and self.verify.has_usable_content(sanitized, request=request):
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

            if (
                allow_needle_correct
                and diagnosis.kind == "degenerate_tool_args"
                and self.needle_correct_degenerate
                and self.tool_path is not None
                and self.tool_path.enabled_for(request)
            ):
                corrected = await self._needle_correct_degenerate(
                    request, diagnosis, sanitized, session, attempt
                )
                if corrected is not None:
                    current = corrected
                    repaired = True
                    continue

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

    async def _needle_correct_degenerate(
        self,
        request: dict[str, Any],
        diagnosis: Any,
        sanitized: dict[str, Any],
        session: TraceSession,
        attempt: int,
    ) -> dict[str, Any] | None:
        del sanitized
        detail = diagnosis.detail or {}
        task = original_user_text(request)
        instruction = correction_instruction(
            task=task,
            tool=detail.get("tool"),
            reason=detail.get("reason"),
            path=detail.get("path"),
            failed_preview=str(detail),
        )
        session.event(
            "repair",
            "ok",
            phase="needle_correct_start",
            attempt=attempt + 1,
            policy_id=self.tool_path.policy_id,
        )
        path = detail.get("path")
        # If Maple NL / task embeds a fenced file for this path, chunk-apply it.
        if self.needle_chunk_writes and isinstance(path, str):
            for fpath, content in extract_fenced_files(task):
                if fpath.replace("\\", "/") == path.replace("\\", "/"):
                    chunked = await asyncio.to_thread(
                        self.tool_path.apply_chunked_file, fpath, content, request
                    )
                    if chunked is not None:
                        session.event(
                            "repair",
                            "ok",
                            phase="needle_correct_done",
                            via=chunked.via,
                            attempt=attempt + 1,
                        )
                        return chunked.response

        result = await asyncio.to_thread(
            partial(
                self.tool_path.handle_instruction,
                instruction,
                request,
                via="needle_correct_degenerate",
            )
        )
        if result is None or result.empty_call:
            session.event(
                "repair",
                "ok",
                phase="needle_correct_miss",
                reason=getattr(self.tool_path, "last_skip_reason", None),
                attempt=attempt + 1,
            )
            return None
        session.event(
            "repair",
            "ok",
            phase="needle_correct_done",
            via=result.via,
            attempt=attempt + 1,
        )
        return result.response

    def _present(self, upstream: dict[str, Any]) -> dict[str, Any]:
        presented = dict(upstream)
        presented["id"] = presented.get("id") or f"chatcmpl-{uuid.uuid4().hex[:24]}"
        presented["object"] = "chat.completion"
        presented["created"] = int(presented.get("created") or time.time())
        presented["model"] = self.public_model_id
        if "choices" not in presented:
            presented["choices"] = []
        return presented
