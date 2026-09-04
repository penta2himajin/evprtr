from __future__ import annotations

import json

import pytest

from compositor.core import Compositor
from compositor.runtimes.needle import NeedleCompleteResult, NeedleToolRuntime
from compositor.tools.path import NeedleToolPath
from compositor.tools.pseudo_tool import (
    apply_pseudo_tool_calls_to_response,
    parse_pseudo_tool_calls,
)
from compositor.trace import TraceStore


def test_parse_pseudo_tool_calls_json_block():
    text = '<tool_call>\n{"name": "ls", "path": ".", "limit": 50}\n</tool_call>'
    calls = parse_pseudo_tool_calls(text)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "ls"
    args = json.loads(calls[0]["function"]["arguments"])
    assert args["path"] == "."
    assert args["limit"] == 50


def test_parse_pseudo_tool_calls_rejects_shell_compound_name():
    text = '<tool_call>\n{"name": "ls -la bench/", "limit": 500}\n</tool_call>'
    assert parse_pseudo_tool_calls(text) == []


def test_apply_pseudo_clears_markup_and_sets_finish_reason():
    content = 'Before\n<tool_call>\n{"name":"read","path":"README.md"}\n</tool_call>\n'
    upstream = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            }
        ]
    }
    calls = parse_pseudo_tool_calls(upstream["choices"][0]["message"]["content"])
    out = apply_pseudo_tool_calls_to_response(upstream, calls)
    msg = out["choices"][0]["message"]
    assert out["choices"][0]["finish_reason"] == "tool_calls"
    assert msg["tool_calls"][0]["function"]["name"] == "read"
    assert "<tool_call" not in (msg.get("content") or "").lower()


@pytest.mark.asyncio
async def test_maple_tools_primary_needle_structures_when_maple_empty(tmp_path):
    """Empty Maple-with-tools → Needle structures from the user task."""

    class Maple:
        async def chat_completions(self, payload):
            # Markup primary: tools become <tools> in system; OpenAI tools popped.
            assert "tools" not in payload
            sys0 = (payload.get("messages") or [{}])[0]
            assert sys0.get("role") == "system"
            assert "<tools>" in (sys0.get("content") or "")
            return {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": ""},
                        "finish_reason": "stop",
                    }
                ]
            }

    class RT(NeedleToolRuntime):
        def __init__(self):
            self.queries: list[str] = []

        def available(self):
            return True

        def complete(self, query, tools, *, max_new_tokens=None):
            self.queries.append(query)
            return NeedleCompleteResult(
                function_calls=[{"name": "ls", "arguments": {"path": "."}}],
                confidence=0.8,
                reasoning="list root",
                raw={},
            )

    rt = RT()
    c = Compositor(
        Maple(),
        traces=TraceStore(tmp_path),
        tool_path=NeedleToolPath(rt),
        maple_tools_primary=True,
        buffer_side_effects=False,
    )
    result = await c.chat_completions(
        {
            "messages": [{"role": "user", "content": "List files in the repo root."}],
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
        }
    )
    assert rt.queries
    msg = result.response["choices"][0]["message"]
    assert msg.get("tool_calls")
    assert msg["tool_calls"][0]["function"]["name"] == "ls"
    phases = [e.detail.get("phase") for e in result.trace.events if isinstance(e.detail, dict)]
    assert "needle_structure_fallback" in phases
    assert result.trace.response_summary.get("needle_via") == "needle_structure_fallback"


@pytest.mark.asyncio
async def test_maple_tools_primary_passes_prose_without_needle(tmp_path):
    class Maple:
        async def chat_completions(self, payload):
            assert "tools" not in payload
            assert "<tools>" in ((payload.get("messages") or [{}])[0].get("content") or "")
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": (
                                "Chibi is a measurement harness for Qwen3.5-4B "
                                "on Apple Silicon M1 Max targeting 150 tok/s."
                            ),
                        },
                        "finish_reason": "stop",
                    }
                ]
            }

    class RT(NeedleToolRuntime):
        def available(self):
            return True

        def complete(self, query, tools, *, max_new_tokens=None):
            raise AssertionError("Needle must not run for prose stop")

    c = Compositor(
        Maple(),
        traces=TraceStore(tmp_path),
        tool_path=NeedleToolPath(RT()),
        needle_via_maple_nl=True,  # ignored when maple_tools_primary
        maple_tools_primary=True,
        buffer_side_effects=False,
    )
    result = await c.chat_completions(
        {
            "messages": [{"role": "user", "content": "Explain the repo."}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "ls",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        }
    )
    choice = result.response["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert "Chibi is a measurement harness" in (choice["message"].get("content") or "")
    phases = [e.detail.get("phase") for e in result.trace.events if isinstance(e.detail, dict)]
    assert "maple_tools_primary" in phases
    assert "maple_final_content" in phases
    assert "maple_nl_start" not in phases
    assert "needle_structure_fallback" not in phases
    assert result.trace.response_summary.get("maple_tools_primary") is True
    assert result.trace.response_summary.get("needle_via") == "maple_final_content"


@pytest.mark.asyncio
async def test_maple_tools_primary_promotes_pseudo_markup(tmp_path):
    class Maple:
        async def chat_completions(self, payload):
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": '<tool_call>\n{"name":"ls","path":"."}\n</tool_call>',
                        },
                        "finish_reason": "stop",
                    }
                ]
            }

    class RT(NeedleToolRuntime):
        def available(self):
            return True

        def complete(self, query, tools, *, max_new_tokens=None):
            raise AssertionError("deterministic parse should win")

    c = Compositor(
        Maple(),
        traces=TraceStore(tmp_path),
        tool_path=NeedleToolPath(RT()),
        maple_tools_primary=True,
        buffer_side_effects=False,
    )
    result = await c.chat_completions(
        {
            "messages": [{"role": "user", "content": "list files"}],
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
        }
    )
    msg = result.response["choices"][0]["message"]
    assert msg.get("tool_calls")
    assert msg["tool_calls"][0]["function"]["name"] == "ls"
    phases = [e.detail.get("phase") for e in result.trace.events if isinstance(e.detail, dict)]
    assert "pseudo_tool_promoted" in phases
