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
from compositor.tools.coerce_read_ls import (
    coerce_ls_file_to_read,
    coerce_read_directory_to_ls,
)
from compositor.tools.convert import openai_tools_to_needle
from compositor.tools.maple_nl import (
    align_create_file_tool_calls,
    correction_instruction,
    extract_fenced_files,
    maple_message_channels,
    maple_nl_request,
    needle_retry_instruction,
    no_tool_call_assistant_content,
    no_tool_call_stop_response,
    parse_create_file,
    prepare_needle_instruction,
    synthetic_mutation_response,
    task_wants_mutation,
    tool_calls_satisfy_mutation,
    user_task_instruction,
)
from compositor.tools.path import NeedleToolPath, ToolPathResult
from compositor.tools.pseudo_tool import (
    apply_pseudo_tool_calls_to_response,
    parse_pseudo_tool_calls,
)
from compositor.trace import (
    FailureLocus,
    TraceRecord,
    TraceSession,
    TraceStore,
    summarize_request,
    summarize_response,
)
from compositor.verify.detectors_tools import DegenerateToolCallDetector
from compositor.verify.pipeline import VerifyBundle, assistant_message
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
        maple_tools_primary: bool | None = None,
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
        # Default: Maple-with-tools first; Needle only structures/repairs misses.
        # Set EVPRTR_MAPLE_TOOLS_PRIMARY=0 to restore Maple-NL→Needle as primary.
        self.maple_tools_primary = (
            _env_flag("EVPRTR_MAPLE_TOOLS_PRIMARY", "1")
            if maple_tools_primary is None
            else maple_tools_primary
        )

    async def chat_completions(self, request: dict[str, Any]) -> ComposeResult:
        session = self.traces.start(
            public_model=self.public_model_id,
            request_summary=summarize_request(request),
        )
        session.event("accept", "ok")

        # Skip when harness forbids tools (tool_choice=none): keep verify/repair.
        if (
            self.maple_tools_primary
            and request.get("tools")
            and request.get("tool_choice", "auto") != "none"
        ):
            return await self._maple_tools_primary_path(request, session)

        # Preferred: Maple NL plan → Needle structures (and optional chunk apply).
        nl_result = await self._maybe_maple_nl_needle(request, session)
        if nl_result is not None:
            return await self._finish_tool_result(nl_result, session, request)

        tool_result = await self._maybe_tool_path(request, session)
        if tool_result is not None:
            return await self._finish_tool_result(tool_result, session, request)

        return await self._maple_with_tools_finish(request, session)

    async def _maple_with_tools_finish(
        self, request: dict[str, Any], session: TraceSession
    ) -> ComposeResult:
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
        upstream = coerce_ls_file_to_read(
            coerce_read_directory_to_ls(
                align_create_file_tool_calls(upstream, original_user_text(request))
            )
        )

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

    async def _maple_tools_primary_path(
        self, request: dict[str, Any], session: TraceSession
    ) -> ComposeResult:
        """Maple-with-tools first; Needle structures/repairs only on misses.

        - Valid ``tool_calls`` → verify (Needle corrects degenerate args).
        - Pseudo ``<tool_call>`` markup → deterministic promote, else Needle.
        - Prose content without tools → stop (do not send essays to Needle).
        - Empty / unusable Maple → Needle structure fallback from task/reasoning.
        """
        session.event(
            "tool_select",
            "ok",
            phase="maple_tools_primary",
            policy_id="maple_tools_primary.v1",
        )
        try:
            session.event("upstream_call", "ok", phase="start", runtime="maple")
            upstream = await self.runtime.chat_completions(request)
            session.event("upstream_call", "ok", phase="done", runtime="maple")
        except UpstreamError as exc:
            session.fail(FailureLocus.UPSTREAM, str(exc), stage="upstream_call")
            session.finish(ok=False)
            raise
        except Exception as exc:  # noqa: BLE001
            session.fail(FailureLocus.UPSTREAM, f"unexpected: {exc}", stage="upstream_call")
            session.finish(ok=False)
            raise

        maple_content, maple_reasoning = maple_message_channels(upstream)
        msg = assistant_message(upstream) or {}
        structured = msg.get("tool_calls") or []
        session.event(
            "tool_select",
            "ok",
            phase="maple_tools_done",
            maple_content=maple_content,
            maple_reasoning=maple_reasoning,
            has_tool_calls=bool(structured),
            presented_tool_calls=_presented_tool_calls(upstream),
            policy_id="maple_tools_primary.v1",
        )

        via = "maple_tools_primary"
        user_task = original_user_text(request)

        if not structured:
            content = msg.get("content") if isinstance(msg.get("content"), str) else ""
            if content and "<tool_call" in content.lower():
                pseudo = parse_pseudo_tool_calls(content)
                if pseudo:
                    upstream = apply_pseudo_tool_calls_to_response(upstream, pseudo)
                    via = "pseudo_tool_promoted"
                    session.event(
                        "tool_select",
                        "ok",
                        phase="pseudo_tool_promoted",
                        n_calls=len(pseudo),
                        presented_tool_calls=_presented_tool_calls(upstream),
                        policy_id="maple_tools_primary.v1",
                    )
                    structured = pseudo
                else:
                    # Broken markup → ask Needle to normalize/structure.
                    fb = await self._needle_structure_fallback(
                        request,
                        session,
                        instruction=content.strip() or maple_reasoning,
                        reason="unparseable_pseudo_tool_markup",
                    )
                    if fb is not None:
                        return await self._finish_tool_result(fb, session, request)

            if (
                not structured
                and len(maple_content.strip()) >= 40
                and "<tool_call" not in maple_content.lower()
            ):
                # Intentional answer — do not Needle-structure prose.
                # If content still has unparseable <tool_call> markup, fall
                # through to verify/repair instead of stopping with the markup.
                session.event(
                    "tool_select",
                    "ok",
                    phase="maple_prose_stop",
                    content_len=len(maple_content),
                    policy_id="maple_tools_primary.v1",
                )
                try:
                    presented = self._present(upstream)
                    presented, buffered_ids = self._buffer_side_effects(
                        presented, session, session.trace_id
                    )
                    session.event("present", "ok", via="maple_prose_stop", prose_stop=True)
                except Exception as exc:  # noqa: BLE001
                    session.fail(FailureLocus.PRESENT, str(exc), stage="present")
                    session.finish(ok=False)
                    raise
                summary = summarize_response(presented)
                summary["repetition_repaired"] = False
                summary["needle_tool_path"] = False
                summary["maple_tools_primary"] = True
                summary["needle_via"] = "maple_prose_stop"
                summary["buffered_approvals"] = buffered_ids
                trace = session.finish(ok=True, response_summary=summary)
                return ComposeResult(response=presented, trace=trace)

            if not structured:
                # Empty / thin Maple → Needle structures from task (or reasoning).
                seed = maple_reasoning.strip() or maple_content.strip() or user_task
                fb = await self._needle_structure_fallback(
                    request,
                    session,
                    instruction=seed,
                    reason="maple_empty_or_thin",
                )
                if fb is not None:
                    return await self._finish_tool_result(fb, session, request)

        # Structured (or last-resort empty) → verify; Needle may correct degenerate.
        upstream, repaired = await self._verify_and_repair(request, upstream, session)
        upstream = coerce_ls_file_to_read(
            coerce_read_directory_to_ls(
                align_create_file_tool_calls(upstream, original_user_text(request))
            )
        )
        try:
            presented = self._present(upstream)
            presented, buffered_ids = self._buffer_side_effects(
                presented, session, session.trace_id
            )
            session.event("present", "ok", via=via, repaired=repaired)
        except Exception as exc:  # noqa: BLE001
            session.fail(FailureLocus.PRESENT, str(exc), stage="present")
            session.finish(ok=False)
            raise
        summary = summarize_response(presented)
        summary["repetition_repaired"] = repaired
        summary["needle_tool_path"] = False
        summary["maple_tools_primary"] = True
        summary["needle_via"] = via
        summary["buffered_approvals"] = buffered_ids
        trace = session.finish(ok=True, response_summary=summary)
        return ComposeResult(response=presented, trace=trace)

    async def _needle_structure_fallback(
        self,
        request: dict[str, Any],
        session: TraceSession,
        *,
        instruction: str,
        reason: str,
    ) -> ToolPathResult | None:
        """Ask Needle to structure a short instruction when Maple missed tools."""
        if self.tool_path is None or not self.tool_path.enabled_for(request):
            return None
        user_task = original_user_text(request)
        tool_names = [
            str(t.get("name"))
            for t in openai_tools_to_needle(request.get("tools"))
            if t.get("name")
        ]
        raw = (instruction or "").strip()
        # Prefer a short imperative; long essays become user-task briefs.
        if len(raw) > 400 or _looks_like_final_prose(raw):
            needle_input = needle_retry_instruction(
                maple_nl=raw,
                user_task=user_task,
                tool_names=tool_names,
            )
            if not needle_input.strip():
                needle_input = user_task_instruction(user_task)
        else:
            needle_input = prepare_needle_instruction(raw, user_task=user_task)

        session.event(
            "tool_select",
            "ok",
            phase="needle_structure_fallback",
            reason=reason,
            needle_instruction=needle_input,
            needle_instruction_len=len(needle_input),
            policy_id="maple_tools_primary.v1",
        )
        result = await asyncio.to_thread(
            self.tool_path.handle_instruction,
            needle_input,
            request,
            via="needle_structure_fallback",
        )
        self._emit_needle_complete(session, attempt="structure_fallback")
        ok, miss = self._nl_needle_acceptable(result, request, user_task)
        if ok and result is not None:
            return self._finalize_nl_result(result, user_task)

        # One allowlisted retry if the first structure pass abstained.
        retry = needle_retry_instruction(
            maple_nl=raw or user_task,
            user_task=user_task,
            tool_names=tool_names,
        )
        if retry.strip() and retry != needle_input:
            session.event(
                "tool_select",
                "ok",
                phase="needle_structure_retry",
                needle_instruction=retry,
                reason=miss or getattr(self.tool_path, "last_skip_reason", None),
                policy_id="maple_tools_primary.v1",
            )
            result = await asyncio.to_thread(
                self.tool_path.handle_instruction,
                retry,
                request,
                via="needle_structure_fallback",
            )
            self._emit_needle_complete(session, attempt="structure_retry")
            ok2, _miss2 = self._nl_needle_acceptable(result, request, user_task)
            if ok2 and result is not None:
                return self._finalize_nl_result(result, user_task)

        session.event(
            "tool_select",
            "ok",
            phase="needle_structure_miss",
            reason=miss or getattr(self.tool_path, "last_skip_reason", None),
            policy_id="maple_tools_primary.v1",
        )
        return None

    async def _finish_tool_result(
        self,
        tool_result: ToolPathResult,
        session: TraceSession,
        request: dict[str, Any],
    ) -> ComposeResult:
        current = tool_result.response
        via = getattr(tool_result, "via", "needle_tool_path")
        try:
            # Full verify loop (Needle correct + re-diagnose + Maple repair).
            # One-shot correct previously presented still-degenerate edits.
            current, _repaired = await self._verify_and_repair(
                request, current, session, allow_needle_correct=True
            )
            current = coerce_ls_file_to_read(
                coerce_read_directory_to_ls(
                    align_create_file_tool_calls(current, original_user_text(request))
                )
            )
            presented = self._present(current)
            presented, buffered_ids = self._buffer_side_effects(
                presented, session, session.trace_id
            )
            session.event("present", "ok", via=via)
        except Exception as exc:  # noqa: BLE001
            session.fail(FailureLocus.PRESENT, str(exc), stage="present")
            session.finish(ok=False)
            raise
        summary = summarize_response(presented)
        summary["repetition_repaired"] = False
        summary["needle_tool_path"] = bool(tool_result.used_needle) or True
        summary["needle_empty_call"] = tool_result.empty_call
        summary["needle_confidence"] = tool_result.confidence
        summary["needle_via"] = via
        summary["buffered_approvals"] = buffered_ids
        if self.maple_tools_primary:
            summary["maple_tools_primary"] = True
        trace = session.finish(ok=True, response_summary=summary)
        return ComposeResult(response=presented, trace=trace)

    def _nl_needle_acceptable(
        self,
        result: ToolPathResult | None,
        request: dict[str, Any],
        user_task: str,
    ) -> tuple[bool, str | None]:
        """Whether an NL→Needle result is good enough to present (or needs retry)."""
        if result is None:
            return False, getattr(self.tool_path, "last_skip_reason", None)
        if result.empty_call:
            return False, "empty_call"
        if task_wants_mutation(user_task) and not tool_calls_satisfy_mutation(result.response):
            return False, "missing_mutation"
        message = assistant_message(result.response)
        if message is None:
            return False, "no_message"
        hit = DegenerateToolCallDetector().diagnose(message, request=request)
        if hit is not None:
            detail = hit.detail or {}
            return False, f"degenerate:{detail.get('reason')}:{detail.get('path')}"
        return True, None

    def _finalize_nl_result(self, result: ToolPathResult, user_task: str) -> ToolPathResult:
        aligned = coerce_ls_file_to_read(
            coerce_read_directory_to_ls(align_create_file_tool_calls(result.response, user_task))
        )
        if aligned is result.response:
            return result
        return ToolPathResult(
            response=aligned,
            used_needle=result.used_needle,
            confidence=result.confidence,
            empty_call=result.empty_call,
            reasoning=result.reasoning,
            via=result.via,
        )

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

        maple_content, maple_reasoning = maple_message_channels(maple_out)
        instruction = maple_content
        maple_nl_source = "content"
        # Prefer content; if Maple stuffed the plan into reasoning only, use that.
        if not instruction:
            if maple_reasoning:
                instruction = maple_reasoning
                maple_nl_source = "reasoning_content"
            else:
                maple_nl_source = "empty"
        user_task = original_user_text(request)
        session.event(
            "tool_select",
            "ok",
            phase="maple_nl_done",
            maple_nl=instruction,
            maple_nl_source=maple_nl_source,
            maple_content=maple_content,
            maple_reasoning=maple_reasoning,
            instruction_len=len(instruction),
            policy_id=self.tool_path.policy_id,
        )
        if not instruction:
            instruction = user_task_instruction(user_task)
            session.event(
                "tool_select",
                "ok",
                phase="maple_nl_empty_use_task",
                maple_nl=instruction,
                policy_id=self.tool_path.policy_id,
            )

        needle_input = prepare_needle_instruction(instruction, user_task=user_task)
        session.event(
            "tool_select",
            "ok",
            phase="needle_instruction_ready",
            needle_instruction=needle_input,
            needle_instruction_len=len(needle_input),
            policy_id=self.tool_path.policy_id,
        )

        if self.needle_chunk_writes:
            fenced = extract_fenced_files(instruction) or extract_fenced_files(needle_input)
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

        result = await asyncio.to_thread(self.tool_path.handle_instruction, needle_input, request)
        self._emit_needle_complete(session, attempt="primary")
        ok, miss_reason = self._nl_needle_acceptable(result, request, user_task)
        if not ok:
            if result is not None:
                session.event(
                    "tool_select",
                    "ok",
                    phase="needle_quality_miss",
                    reason=miss_reason,
                    via=getattr(result, "via", None),
                    policy_id=self.tool_path.policy_id,
                )
            result = None
        if result is None:
            # Intentional no-tool NL (marker) → stop with Maple prose; do not
            # retry Needle or pay for fallback Maple-with-tools.
            stop_content = no_tool_call_assistant_content(instruction, user_task=user_task)
            if stop_content is not None:
                session.event(
                    "tool_select",
                    "ok",
                    phase="final_answer_stop",
                    via="maple_nl_no_tool",
                    content_len=len(stop_content),
                    maple_nl=instruction,
                    reason=miss_reason or getattr(self.tool_path, "last_skip_reason", None),
                    policy_id=self.tool_path.policy_id,
                )
                return ToolPathResult(
                    response=no_tool_call_stop_response(stop_content, model=self.public_model_id),
                    used_needle=False,
                    confidence=None,
                    empty_call=False,
                    reasoning="maple_nl_no_tool_call",
                    via="maple_nl_no_tool",
                )
            # Second chance: allowlisted imperative retry (soft user-task briefs
            # often made Needle invent phone-call / missing-write refusals).
            tool_names = [
                str(t.get("name"))
                for t in openai_tools_to_needle(request.get("tools"))
                if t.get("name")
            ]
            retry = needle_retry_instruction(
                maple_nl=instruction,
                user_task=user_task,
                tool_names=tool_names,
            )
            if retry.strip() and retry != needle_input:
                session.event(
                    "tool_select",
                    "ok",
                    phase="needle_retry_user_task",
                    needle_instruction=retry,
                    needle_instruction_len=len(retry),
                    reason=miss_reason or getattr(self.tool_path, "last_skip_reason", None),
                    policy_id=self.tool_path.policy_id,
                )
                result = await asyncio.to_thread(
                    self.tool_path.handle_instruction,
                    retry,
                    request,
                )
                self._emit_needle_complete(session, attempt="retry")
                ok2, miss2 = self._nl_needle_acceptable(result, request, user_task)
                if not ok2:
                    session.event(
                        "tool_select",
                        "ok",
                        phase="needle_retry_still_bad",
                        reason=miss2,
                        policy_id=self.tool_path.policy_id,
                    )
                    result = None
                    # Retry may still land on no-tool NL (same instruction).
                    stop_content = no_tool_call_assistant_content(instruction, user_task=user_task)
                    if stop_content is not None:
                        session.event(
                            "tool_select",
                            "ok",
                            phase="final_answer_stop",
                            via="maple_nl_no_tool",
                            content_len=len(stop_content),
                            maple_nl=instruction,
                            reason=miss2,
                            policy_id=self.tool_path.policy_id,
                        )
                        return ToolPathResult(
                            response=no_tool_call_stop_response(
                                stop_content, model=self.public_model_id
                            ),
                            used_needle=False,
                            confidence=None,
                            empty_call=False,
                            reasoning="maple_nl_no_tool_call",
                            via="maple_nl_no_tool",
                        )
        if result is None:
            synthetic = synthetic_mutation_response(user_task)
            if synthetic is not None:
                via = (
                    "synthetic_create_file"
                    if parse_create_file(user_task)
                    else "synthetic_edit_replace"
                )
                session.event(
                    "tool_select",
                    "ok",
                    phase="done",
                    via=via,
                    policy_id=self.tool_path.policy_id,
                )
                return ToolPathResult(
                    response=synthetic,
                    used_needle=False,
                    confidence=None,
                    empty_call=False,
                    reasoning="deterministic_mutation",
                    via=via,
                )
            session.event(
                "tool_select",
                "ok",
                phase="fallback_maple",
                reason=miss_reason or getattr(self.tool_path, "last_skip_reason", None),
                policy_id=self.tool_path.policy_id,
            )
            return None
        result = self._finalize_nl_result(result, user_task)
        session.event(
            "tool_select",
            "ok",
            phase="done",
            via=result.via,
            empty_call=result.empty_call,
            confidence=result.confidence,
            presented_tool_calls=_presented_tool_calls(result.response),
            policy_id=self.tool_path.policy_id,
        )
        return result

    def _emit_needle_complete(self, session: TraceSession, *, attempt: str) -> None:
        if self.tool_path is None:
            return
        detail = self.tool_path.pop_last_complete()
        if not detail:
            return
        session.event(
            "tool_select",
            "ok",
            phase="needle_complete",
            attempt=attempt,
            policy_id=self.tool_path.policy_id,
            **detail,
        )

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
                "missing_mutation",
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
                and diagnosis.kind in {"degenerate_tool_args", "missing_mutation"}
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
        # Prefer imperative create-file / mutation briefs; soft correction alone
        # often yields another unusable edit (live smoke4).
        candidates: list[str] = []
        if diagnosis.kind == "missing_mutation" or parse_create_file(task) is not None:
            candidates.append(user_task_instruction(task))
        else:
            candidates.append(
                correction_instruction(
                    task=task,
                    tool=detail.get("tool"),
                    reason=detail.get("reason"),
                    path=detail.get("path"),
                    failed_preview=str(detail),
                )
            )
            if task_wants_mutation(task):
                ut = user_task_instruction(task)
                if ut not in candidates:
                    candidates.append(ut)

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
                        aligned = coerce_ls_file_to_read(
                            coerce_read_directory_to_ls(
                                align_create_file_tool_calls(chunked.response, task)
                            )
                        )
                        ok, _ = self._nl_needle_acceptable(
                            ToolPathResult(
                                response=aligned,
                                used_needle=chunked.used_needle,
                                confidence=chunked.confidence,
                                empty_call=False,
                                reasoning=chunked.reasoning,
                                via=chunked.via,
                            ),
                            request,
                            task,
                        )
                        if ok:
                            session.event(
                                "repair",
                                "ok",
                                phase="needle_correct_done",
                                via=chunked.via,
                                attempt=attempt + 1,
                            )
                            return aligned

        for instruction in candidates:
            result = await asyncio.to_thread(
                partial(
                    self.tool_path.handle_instruction,
                    instruction,
                    request,
                    via="needle_correct_degenerate",
                )
            )
            if result is None or result.empty_call:
                continue
            aligned = coerce_ls_file_to_read(
                coerce_read_directory_to_ls(align_create_file_tool_calls(result.response, task))
            )
            shaped = ToolPathResult(
                response=aligned,
                used_needle=result.used_needle,
                confidence=result.confidence,
                empty_call=result.empty_call,
                reasoning=result.reasoning,
                via=result.via,
            )
            ok, reason = self._nl_needle_acceptable(shaped, request, task)
            if not ok:
                session.event(
                    "repair",
                    "ok",
                    phase="needle_correct_still_bad",
                    reason=reason,
                    attempt=attempt + 1,
                )
                continue
            session.event(
                "repair",
                "ok",
                phase="needle_correct_done",
                via=result.via,
                attempt=attempt + 1,
            )
            return aligned

        synthetic = synthetic_mutation_response(task)
        if synthetic is not None:
            via = "synthetic_create_file" if parse_create_file(task) else "synthetic_edit_replace"
            session.event(
                "repair",
                "ok",
                phase="needle_correct_done",
                via=via,
                attempt=attempt + 1,
            )
            return synthetic

        session.event(
            "repair",
            "ok",
            phase="needle_correct_miss",
            reason=getattr(self.tool_path, "last_skip_reason", None),
            attempt=attempt + 1,
        )
        return None

    def _present(self, upstream: dict[str, Any]) -> dict[str, Any]:
        presented = dict(upstream)
        presented["id"] = presented.get("id") or f"chatcmpl-{uuid.uuid4().hex[:24]}"
        presented["object"] = "chat.completion"
        presented["created"] = int(presented.get("created") or time.time())
        presented["model"] = self.public_model_id
        if "choices" not in presented:
            presented["choices"] = []
        return presented


def _presented_tool_calls(response: dict[str, Any]) -> list[dict[str, Any]]:
    """Compact tool_calls from a chat.completion for traces."""
    choices = response.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return []
    message = choices[0].get("message") or {}
    calls = message.get("tool_calls") or []
    out: list[dict[str, Any]] = []
    for call in calls:
        if not isinstance(call, dict):
            continue
        fn = call.get("function") if isinstance(call.get("function"), dict) else {}
        args = fn.get("arguments")
        out.append(
            {
                "name": fn.get("name"),
                "arguments": args,
            }
        )
    return out


def _looks_like_final_prose(text: str) -> bool:
    """Heuristic: essay/summary that should not be fed raw to Needle."""
    low = (text or "").lower()
    if len(low) < 120:
        return False
    hints = (
        "summary",
        "overview",
        "this repository",
        "this project",
        "based on my",
        "highlights",
        "in conclusion",
        "## ",
    )
    return any(h in low for h in hints)
