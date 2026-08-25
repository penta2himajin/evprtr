from __future__ import annotations

import json

import pytest

from compositor.core import Compositor
from compositor.runtimes.needle import NeedleCompleteResult, NeedleToolRuntime
from compositor.tools.chunk_write import CHUNK_SENTINEL, planned_tool_calls, split_content
from compositor.tools.maple_nl import (
    extract_fenced_files,
    maple_nl_request,
)
from compositor.tools.path import NeedleToolPath
from compositor.trace import TraceStore


def test_split_content_line_aware():
    text = "a\n" * 50 + "b\n" * 50
    parts = split_content(text, max_chars=40)
    assert len(parts) > 1
    assert "".join(parts) == text


def test_planned_tool_calls_multi_chunk():
    calls = planned_tool_calls("Cargo.toml", ["one\n", "two\n", "three\n"])
    assert len(calls) == 3
    assert calls[0]["function"]["name"] == "write"
    args0 = json.loads(calls[0]["function"]["arguments"])
    assert args0["content"].endswith(CHUNK_SENTINEL)
    assert calls[1]["function"]["name"] == "edit"
    assert calls[2]["function"]["name"] == "edit"
    last = json.loads(calls[2]["function"]["arguments"])
    assert last["edits"][0]["newText"] == "three\n"
    assert CHUNK_SENTINEL not in last["edits"][0]["newText"]


def test_extract_fenced_files():
    text = (
        "Please write Cargo.toml:\n"
        "```Cargo.toml\n"
        '[package]\nname = "tinyserve"\n'
        "```\n"
    )
    files = extract_fenced_files(text)
    assert files
    assert files[0][0] == "Cargo.toml"
    assert "tinyserve" in files[0][1]


def test_maple_nl_request_strips_tools():
    req = maple_nl_request(
        {
            "messages": [{"role": "user", "content": "add axum"}],
            "tools": [{"type": "function", "function": {"name": "write"}}],
            "tool_choice": "auto",
        }
    )
    assert "tools" not in req
    assert req["tool_choice"] == "none"
    assert req["messages"][0]["role"] == "system"


@pytest.mark.asyncio
async def test_maple_nl_then_needle(tmp_path):
    class Maple:
        async def chat_completions(self, payload):
            assert payload.get("tool_choice") == "none"
            assert "tools" not in payload
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": (
                                "Use write on Cargo.toml:\n"
                                "```Cargo.toml\n"
                                'axum = "0.7"\n'
                                "```\n"
                            ),
                        }
                    }
                ]
            }

    class RT(NeedleToolRuntime):
        def available(self):
            return True

        def complete(self, query, tools, *, max_new_tokens=None):
            return NeedleCompleteResult(
                function_calls=[],
                confidence=0.1,
                reasoning="abstain",
                raw={},
            )

    c = Compositor(
        Maple(),
        traces=TraceStore(tmp_path),
        tool_path=NeedleToolPath(RT(), prefer_chunk_writes=True, chunk_chars=20),
        needle_via_maple_nl=True,
        needle_chunk_writes=True,
        buffer_side_effects=False,
    )
    result = await c.chat_completions(
        {
            "messages": [{"role": "user", "content": "Add axum to Cargo.toml"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "write",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"},
                                "content": {"type": "string"},
                            },
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "edit",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
            ],
        }
    )
    msg = result.response["choices"][0]["message"]
    assert msg.get("tool_calls")
    assert result.trace.response_summary.get("needle_via")


@pytest.mark.asyncio
async def test_needle_corrects_degenerate_write(tmp_path):
    class Maple:
        async def chat_completions(self, payload):
            # Agent-with-tools path after NL miss — emit degenerate write.
            if payload.get("tool_choice") == "none":
                return {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "please write the file",
                            }
                        }
                    ]
                }
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_bad",
                                    "type": "function",
                                    "function": {
                                        "name": "write",
                                        "arguments": json.dumps(
                                            {"path": "Tower", "content": "0.5"}
                                        ),
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            }

    class RT(NeedleToolRuntime):
        def available(self):
            return True

        def complete(self, query, tools, *, max_new_tokens=None):
            if "unusable" in query.lower() or "Failed tool" in query:
                return NeedleCompleteResult(
                    function_calls=[
                        {
                            "name": "write",
                            "arguments": {
                                "path": "Cargo.toml",
                                "content": '[package]\nname="x"\naxum="0.7"\n',
                            },
                        }
                    ],
                    confidence=0.9,
                    reasoning=None,
                    raw={},
                )
            return NeedleCompleteResult([], 0.01, "nope", {})

    c = Compositor(
        Maple(),
        traces=TraceStore(tmp_path),
        tool_path=NeedleToolPath(RT()),
        needle_via_maple_nl=True,
        needle_correct_degenerate=True,
        needle_chunk_writes=False,
        buffer_side_effects=False,
        repair_attempts=2,
    )
    result = await c.chat_completions(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Write Cargo.toml with axum dependency please now.",
                }
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "write",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"},
                                "content": {"type": "string"},
                            },
                        },
                    },
                }
            ],
        }
    )
    # NL→Needle abstains → Maple tools emits Tower → correct to Cargo.toml
    args = result.response["choices"][0]["message"]["tool_calls"][0]["function"][
        "arguments"
    ]
    if isinstance(args, str):
        args = json.loads(args)
    assert args["path"] == "Cargo.toml"
    assert "axum" in args["content"]
