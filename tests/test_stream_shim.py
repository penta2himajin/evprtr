from __future__ import annotations

from compositor.api.stream_shim import completion_to_sse_chunks


def test_completion_to_sse_chunks_content():
    lines = completion_to_sse_chunks(
        {
            "id": "chatcmpl-x",
            "created": 42,
            "model": "evprtr",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "plan"},
                    "finish_reason": "stop",
                }
            ],
        }
    )
    assert len(lines) == 3
    assert lines[0].startswith("data: ")
    assert '"content": "plan"' in lines[0]
    assert '"finish_reason": null' in lines[0]
    assert '"finish_reason": "stop"' in lines[1]
    assert lines[2] == "data: [DONE]"


def test_completion_to_sse_chunks_tool_calls():
    lines = completion_to_sse_chunks(
        {
            "id": "chatcmpl-y",
            "created": 1,
            "model": "evprtr",
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
                                    "name": "read",
                                    "arguments": '{"path":"AGENTS.md"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }
    )
    assert '"name": "read"' in lines[0]
    assert '"finish_reason": "tool_calls"' in lines[1]
