"""Grammar / structured_outputs must survive compositor paths (Needle off)."""

from __future__ import annotations

from typing import Any

import pytest

from compositor.core import Compositor
from compositor.tools.maple_nl import maple_nl_request
from compositor.trace import TraceStore


def test_maple_nl_request_preserves_structured_outputs():
    req = {
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"type": "function", "function": {"name": "read", "parameters": {}}}],
        "structured_outputs": {"grammar": 'root ::= "ZQX"'},
        "guided_grammar": 'root ::= "ZQX"',
    }
    out = maple_nl_request(req)
    assert "tools" not in out
    assert out["structured_outputs"] == {"grammar": 'root ::= "ZQX"'}
    assert out["guided_grammar"] == 'root ::= "ZQX"'


@pytest.mark.asyncio
async def test_compositor_forwards_structured_outputs_with_needle_disabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
):
    monkeypatch.setenv("EVPRTR_NEEDLE_ENABLED", "0")
    monkeypatch.setenv("EVPRTR_MAPLE_TOOLS_PRIMARY", "1")
    seen: dict[str, Any] = {}

    class CaptureRuntime:
        async def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            seen["payload"] = payload
            return {
                "id": "chatcmpl-gbnf",
                "object": "chat.completion",
                "created": 1,
                "model": "upstream",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "ls",
                                        "arguments": '{"path":"."}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }

        async def aclose(self) -> None:
            return None

    c = Compositor(
        runtime=CaptureRuntime(),
        public_model_id="evprtr",
        tool_path=None,
        traces=TraceStore(tmp_path),
    )
    result = await c.chat_completions(
        {
            "messages": [{"role": "user", "content": "list root"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "ls",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                        },
                    },
                }
            ],
            "tool_choice": "auto",
            "structured_outputs": {"grammar": 'root ::= "ignored-by-tools-path"'},
            "guided_grammar": 'root ::= "also-forward"',
        }
    )
    assert result.response["choices"][0]["message"]["tool_calls"]
    assert seen["payload"]["structured_outputs"] == {"grammar": 'root ::= "ignored-by-tools-path"'}
    assert seen["payload"]["guided_grammar"] == 'root ::= "also-forward"'
