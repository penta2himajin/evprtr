from __future__ import annotations

from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from compositor.api import build_app
from compositor.core import Compositor


class FakeRuntime:
    def __init__(self, response: dict[str, Any] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.response = response or {
            "id": "chatcmpl-upstream",
            "object": "chat.completion",
            "created": 1,
            "model": "maple-upstream",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "hello from upstream"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        }

    async def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(payload)
        return dict(self.response)


@pytest.fixture
def fake_runtime() -> FakeRuntime:
    return FakeRuntime()


@pytest.fixture
async def client(fake_runtime: FakeRuntime, tmp_path):
    from compositor.trace import TraceStore

    store = TraceStore(tmp_path)
    app = build_app(Compositor(fake_runtime, public_model_id="evprtr", traces=store))
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac, fake_runtime, store


@pytest.mark.asyncio
async def test_lists_public_model(client):
    ac, _, _ = client
    response = await ac.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert data["data"][0]["id"] == "evprtr"


@pytest.mark.asyncio
async def test_chat_completions_passthrough_rewrites_model_id(client):
    ac, runtime, store = client
    response = await ac.post(
        "/v1/chat/completions",
        json={
            "model": "evprtr",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "evprtr"
    assert body["choices"][0]["message"]["content"] == "hello from upstream"
    assert runtime.calls[0]["messages"][0]["content"] == "hi"
    trace_id = response.headers.get("x-evprtr-trace-id")
    assert trace_id
    assert store.get(trace_id) is not None


@pytest.mark.asyncio
async def test_stream_shim_returns_sse_from_full_completion(client):
    ac, runtime, store = client
    response = await ac.post(
        "/v1/chat/completions",
        json={
            "model": "evprtr",
            "stream": True,
            "messages": [
                {"role": "developer", "content": "be brief"},
                {"role": "user", "content": "hi"},
            ],
        },
    )
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    text = response.text
    assert "chat.completion.chunk" in text
    assert "hello from upstream" in text
    assert "data: [DONE]" in text
    assert runtime.calls[0]["stream"] is False
    assert runtime.calls[0]["messages"][0]["role"] == "system"
    trace_id = response.headers.get("x-evprtr-trace-id")
    assert trace_id
    assert store.get(trace_id) is not None
    assert store.get(trace_id).ok is True


@pytest.mark.asyncio
async def test_missing_upstream_returns_502_with_trace(tmp_path):
    from compositor.trace import TraceStore

    app = build_app(traces=TraceStore(tmp_path))
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post(
            "/v1/chat/completions",
            json={"model": "evprtr", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert response.status_code == 502
    detail = response.json()["detail"]
    assert detail["locus"] == "upstream"
    assert detail["trace_id"]


@pytest.mark.asyncio
async def test_get_trace_endpoint(client):
    ac, _, _ = client
    posted = await ac.post(
        "/v1/chat/completions",
        json={"model": "evprtr", "messages": [{"role": "user", "content": "hi"}]},
    )
    trace_id = posted.headers["x-evprtr-trace-id"]
    response = await ac.get(f"/v1/traces/{trace_id}")
    assert response.status_code == 200
    assert response.json()["trace_id"] == trace_id
    assert response.json()["ok"] is True


@pytest.mark.asyncio
async def test_enqueue_approval_via_post(client):
    ac, _, _ = client
    response = await ac.post(
        "/v1/approvals",
        json={
            "tool_name": "bash",
            "arguments": '{"command":"echo hi"}',
            "reason": "pi gate audit",
            "tags": ["pi_gate"],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["id"].startswith("appr-")
    assert body["tool_name"] == "bash"
    assert body["status"] == "pending"
    listed = await ac.get("/v1/approvals", params={"status": "pending"})
    assert any(item["id"] == body["id"] for item in listed.json()["data"])
