"""FastAPI app: OpenAI-compatible surface for harnesses."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from compositor.api.schemas import ChatCompletionRequest, ModelCard, ModelList
from compositor.approvals.buffer import SideEffectBuffer
from compositor.approvals.store import ApprovalStatus, ApprovalStore
from compositor.core import Compositor
from compositor.runtimes import UpstreamError
from compositor.runtimes.needle import NeedleToolRuntime
from compositor.runtimes.openai_compat import OpenAICompatRuntime
from compositor.tools.path import NeedleToolPath
from compositor.trace import FailureLocus, TraceStore, summarize_request

TRACE_HEADER = "X-Evprtr-Trace-Id"
APPROVALS_HEADER = "X-Evprtr-Approvals"


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value


class ApprovalDecisionBody(BaseModel):
    note: str | None = None


def build_app(
    compositor: Compositor | None = None,
    *,
    public_model_id: str | None = None,
    traces: TraceStore | None = None,
    approvals: ApprovalStore | None = None,
) -> FastAPI:
    model_id = public_model_id or _env("EVPRTR_MODEL_ID", "evprtr") or "evprtr"

    if compositor is None:
        if traces is None:
            trace_dir = _env("EVPRTR_TRACE_DIR", str(Path(".evprtr") / "traces"))
            traces = TraceStore(trace_dir)
        if approvals is None:
            appr_dir = _env("EVPRTR_APPROVAL_DIR", str(Path(".evprtr") / "approvals"))
            approvals = ApprovalStore(appr_dir)
        upstream = _env("EVPRTR_UPSTREAM_BASE_URL")
        if not upstream:
            runtime = _MissingUpstream()
        else:
            runtime = OpenAICompatRuntime(
                upstream,
                upstream_model=_env("EVPRTR_UPSTREAM_MODEL"),
            )
        tool_path = _maybe_needle_tool_path(model_id)
        buffer_flag = (_env("EVPRTR_BUFFER_SIDE_EFFECTS", "1") or "1").lower()
        buffer_on = buffer_flag not in {"0", "false", "off", "no"}
        compositor = Compositor(
            runtime,
            public_model_id=model_id,
            traces=traces,
            tool_path=tool_path,
            approvals=approvals,
            side_effect_buffer=SideEffectBuffer(approvals, enabled=buffer_on),
            buffer_side_effects=buffer_on,
        )
    else:
        if traces is not None:
            compositor.traces = traces
        if approvals is not None:
            compositor.approvals = approvals
            compositor.side_effect_buffer.store = approvals

    app = FastAPI(title="evprtr", version="0.1.0")
    app.state.compositor = compositor
    app.state.public_model_id = model_id
    app.state.traces = compositor.traces
    app.state.approvals = compositor.approvals

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        needle_on = bool(
            getattr(app.state.compositor, "tool_path", None)
            and app.state.compositor.tool_path.runtime.available()
        )
        return {
            "status": "ok",
            "needle_tool_path": needle_on,
            "buffer_side_effects": app.state.compositor.side_effect_buffer.enabled,
        }

    @app.get("/v1/models")
    async def list_models() -> ModelList:
        return ModelList(data=[ModelCard(id=app.state.public_model_id)])

    @app.get("/v1/traces")
    async def list_traces(limit: int = 20) -> dict[str, Any]:
        records = app.state.traces.list_recent(max(1, min(limit, 100)))
        return {"object": "list", "data": [r.to_dict() for r in records]}

    @app.get("/v1/traces/{trace_id}")
    async def get_trace(trace_id: str) -> dict[str, Any]:
        record = app.state.traces.get(trace_id)
        if record is None:
            raise HTTPException(status_code=404, detail="trace not found")
        return record.to_dict()

    @app.get("/v1/approvals")
    async def list_approvals(status: str | None = None, limit: int = 50) -> dict[str, Any]:
        items = app.state.approvals.list(status=status, limit=limit)
        return {"object": "list", "data": [a.to_dict() for a in items]}

    @app.get("/v1/approvals/{action_id}")
    async def get_approval(action_id: str) -> dict[str, Any]:
        action = app.state.approvals.get(action_id)
        if action is None:
            raise HTTPException(status_code=404, detail="approval not found")
        return action.to_dict()

    @app.post("/v1/approvals/{action_id}/approve")
    async def approve_action(
        action_id: str, body: ApprovalDecisionBody | None = None
    ) -> dict[str, Any]:
        action = app.state.approvals.decide(
            action_id, approve=True, note=(body.note if body else None)
        )
        if action is None:
            raise HTTPException(status_code=404, detail="approval not found")
        return action.to_dict()

    @app.post("/v1/approvals/{action_id}/reject")
    async def reject_action(
        action_id: str, body: ApprovalDecisionBody | None = None
    ) -> dict[str, Any]:
        action = app.state.approvals.decide(
            action_id, approve=False, note=(body.note if body else None)
        )
        if action is None:
            raise HTTPException(status_code=404, detail="approval not found")
        return action.to_dict()

    @app.post("/v1/approvals/{action_id}/apply")
    async def apply_action(action_id: str) -> dict[str, Any]:
        """Mark an approved action as applied.

        v0 does not execute bash/write on the host — apply only records intent.
        Real sandboxed/host apply lands in a later peelable executor.
        """
        action = app.state.approvals.get(action_id)
        if action is None:
            raise HTTPException(status_code=404, detail="approval not found")
        if action.status != ApprovalStatus.APPROVED.value:
            raise HTTPException(
                status_code=409,
                detail=f"action status must be approved (got {action.status})",
            )
        updated = app.state.approvals.mark_applied(
            action_id,
            {
                "executed": False,
                "note": ("v0 apply records approval only; host/sandbox execution not enabled"),
            },
        )
        assert updated is not None
        return updated.to_dict()

    @app.post("/v1/chat/completions")
    async def chat_completions(body: ChatCompletionRequest, request: Request) -> JSONResponse:
        if body.stream:
            session = app.state.traces.start(
                public_model=app.state.public_model_id,
                request_summary=summarize_request(body.model_dump(exclude_none=True)),
            )
            session.fail(
                FailureLocus.ACCEPT,
                "streaming is not implemented in compositor v0; set stream=false",
                stage="accept",
            )
            trace = session.finish(ok=False)
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "streaming is not implemented in compositor v0; set stream=false",
                    "trace_id": trace.trace_id,
                    "locus": trace.locus,
                },
                headers={TRACE_HEADER: trace.trace_id},
            )

        payload = body.model_dump(exclude_none=True)
        raw: dict[str, Any] = await request.json()
        for key, value in raw.items():
            if key not in payload:
                payload[key] = value

        try:
            result = await app.state.compositor.chat_completions(payload)
        except UpstreamError as exc:
            recent = app.state.traces.list_recent(1)
            trace_id = recent[0].trace_id if recent else ""
            locus = recent[0].locus if recent else FailureLocus.UPSTREAM.value
            raise HTTPException(
                status_code=502,
                detail={
                    "error": str(exc),
                    "upstream_status": exc.status_code,
                    "upstream_body": exc.body,
                    "trace_id": trace_id,
                    "locus": locus,
                },
                headers={TRACE_HEADER: trace_id} if trace_id else None,
            ) from exc

        headers = {TRACE_HEADER: result.trace.trace_id}
        buffered = (result.trace.response_summary or {}).get("buffered_approvals") or []
        if buffered:
            headers[APPROVALS_HEADER] = ",".join(buffered)
        return JSONResponse(result.response, headers=headers)

    return app


def _maybe_needle_tool_path(public_model_id: str) -> NeedleToolPath | None:
    flag = (_env("EVPRTR_NEEDLE_ENABLED", "1") or "1").lower()
    if flag in {"0", "false", "off", "no"}:
        return None
    lib = _env("EVPRTR_NEEDLE_LIB_PATH") or _env("NEEDLE_LIB_PATH")
    if not lib:
        candidate = Path("F:/LLM/models/needle2/libneedle.dll")
        if candidate.is_file():
            lib = str(candidate)
    runtime = NeedleToolRuntime(lib_path=lib)
    if not runtime.available():
        return None
    min_conf = float(_env("EVPRTR_NEEDLE_MIN_CONFIDENCE", "0") or "0")
    return NeedleToolPath(
        runtime,
        min_confidence=min_conf,
        public_model_id=public_model_id,
    )


class _MissingUpstream:
    async def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
        raise UpstreamError(
            "EVPRTR_UPSTREAM_BASE_URL is not set; compositor has no runtime to call",
        )
