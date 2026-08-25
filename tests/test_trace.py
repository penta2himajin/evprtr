from __future__ import annotations

from pathlib import Path

import pytest

from compositor.trace import FailureLocus, TraceStore, summarize_request


def test_trace_records_upstream_failure(tmp_path: Path):
    store = TraceStore(tmp_path)
    session = store.start(public_model="evprtr", request_summary={"message_count": 1})
    session.event("accept", "ok")
    session.fail(FailureLocus.UPSTREAM, "connection refused", stage="upstream_call")
    record = session.finish(ok=False)

    assert record.ok is False
    assert record.locus == "upstream"
    assert record.error == "connection refused"
    assert (tmp_path / f"{record.trace_id}.json").is_file()
    loaded = store.get(record.trace_id)
    assert loaded is not None
    assert loaded.locus == "upstream"
    assert any(e.status == "error" for e in loaded.events)


def test_summarize_request_omits_message_bodies():
    summary = summarize_request(
        {
            "model": "evprtr",
            "messages": [
                {"role": "user", "content": "SECRET_SHOULD_NOT_APPEAR"},
            ],
            "tools": [{"type": "function"}],
        }
    )
    assert summary["message_count"] == 1
    assert summary["roles"] == ["user"]
    assert summary["has_tools"] is True
    blob = str(summary)
    assert "SECRET_SHOULD_NOT_APPEAR" not in blob


@pytest.mark.asyncio
async def test_compositor_attaches_trace_on_success(tmp_path: Path):
    from compositor.core import Compositor

    class OkRuntime:
        async def chat_completions(self, payload):
            return {
                "model": "maple",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }

    store = TraceStore(tmp_path)
    c = Compositor(OkRuntime(), public_model_id="evprtr", traces=store)
    result = await c.chat_completions(
        {"model": "evprtr", "messages": [{"role": "user", "content": "hi"}]}
    )
    assert result.response["model"] == "evprtr"
    assert result.trace.ok is True
    assert result.trace.locus == "none"
    assert store.get(result.trace.trace_id) is not None


@pytest.mark.asyncio
async def test_compositor_marks_upstream_locus(tmp_path: Path):
    from compositor.core import Compositor
    from compositor.runtimes import UpstreamError

    class Boom:
        async def chat_completions(self, payload):
            raise UpstreamError("boom", status_code=500, body="nope")

    store = TraceStore(tmp_path)
    c = Compositor(Boom(), public_model_id="evprtr", traces=store)
    with pytest.raises(UpstreamError):
        await c.chat_completions({"messages": []})
    recent = store.list_recent(1)
    assert recent
    assert recent[0].ok is False
    assert recent[0].locus == "upstream"
