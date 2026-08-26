from __future__ import annotations

from typing import Any

import httpx
import pytest

from compositor.runtimes.openai_compat import OpenAICompatRuntime


def _ok_body() -> dict[str, Any]:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": "upstream-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }
        ],
    }


@pytest.mark.asyncio
async def test_sends_authorization_bearer_when_api_key_set():
    seen: dict[str, str | None] = {"authorization": None}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["authorization"] = request.headers.get("Authorization")
        return httpx.Response(200, json=_ok_body())

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    runtime = OpenAICompatRuntime(
        "http://upstream.test/v1",
        upstream_model="upstream-model",
        api_key="sk-test-key",
        client=client,
    )
    try:
        result = await runtime.chat_completions(
            {"model": "ignored", "messages": [{"role": "user", "content": "hi"}]}
        )
    finally:
        await runtime.aclose()

    assert result["choices"][0]["message"]["content"] == "ok"
    assert seen["authorization"] == "Bearer sk-test-key"


@pytest.mark.asyncio
async def test_omits_authorization_when_api_key_unset():
    seen: dict[str, bool] = {"has_authorization": False}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["has_authorization"] = "Authorization" in request.headers
        return httpx.Response(200, json=_ok_body())

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    runtime = OpenAICompatRuntime("http://upstream.test/v1", client=client)
    try:
        await runtime.chat_completions(
            {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
        )
    finally:
        await runtime.aclose()

    assert seen["has_authorization"] is False
