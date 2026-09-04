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
    text = 'Please write Cargo.toml:\n```Cargo.toml\n[package]\nname = "tinyserve"\n```\n'
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
    assert "No tool call needed." in req["messages"][0]["content"]
    assert req["max_tokens"] == 384


def test_prepare_needle_instruction_falls_back_on_degenerate():
    from compositor.tools.maple_nl import prepare_needle_instruction

    bad = "smoke-smoke-smoke-" * 200
    out = prepare_needle_instruction(bad, user_task="Create file x.txt with content hi")
    assert "Create file x.txt" in out
    assert len(out) < len(bad)


def test_is_no_tool_call_nl_requires_explicit_marker():
    from compositor.tools.maple_nl import is_no_tool_call_nl, strip_no_tool_call_marker

    assert is_no_tool_call_nl(
        "No tool call needed.\n\nThis repo is a bench harness for Qwen on M1 Max."
    )
    assert is_no_tool_call_nl(
        "No tool call needed. Chibi measures speculative decoding on Apple Silicon."
    )
    assert not is_no_tool_call_nl('Call ls. path="."')
    assert not is_no_tool_call_nl(
        "This repository is a measurement harness for Qwen3.5-4B on M1 Max."
    )
    assert not is_no_tool_call_nl('No tool call needed.\n<tool_call>\n{"name":"ls"}\n</tool_call>')
    assert not is_no_tool_call_nl("No tool call needed.")
    body = strip_no_tool_call_marker(
        "No tool call needed.\n\nThis repo is a bench harness for Qwen on M1 Max."
    )
    assert body.startswith("This repo is a bench")
    assert "No tool call needed" not in body


def test_no_tool_call_assistant_content_allows_intentional_stop():
    from compositor.tools.maple_nl import no_tool_call_assistant_content

    nl = "No tool call needed.\n\nI would create the file but here is a summary instead."
    # Mutation keyword in the user task no longer blocks an explicit NL stop.
    body = no_tool_call_assistant_content(
        nl, user_task="Create file x.txt with content: hi"
    )
    assert body is not None
    assert body.startswith("I would create")
    assert no_tool_call_assistant_content(
        nl,
        user_task="Explore this repository and explain what it does.",
    ).startswith("I would create")


def test_needle_retry_instruction_is_short_imperative_like_official_examples():
    from compositor.tools.maple_nl import needle_retry_instruction

    out = needle_retry_instruction(
        maple_nl=(
            "List the repo root with ls, then read README.md\n"
            "Also explain the project afterwards in detail.\n"
        ),
        user_task="このリポジトリの内容を調べて",
        tool_names=["read", "ls", "grep", "find"],
    )
    assert out.startswith("Call ls.")
    assert "path=" in out
    assert "phone" not in out.lower()
    assert "Do not invent" not in out
    assert "Planner note:" not in out
    assert "User task:" not in out
    assert "sms" not in out.lower()
    assert len(out) < 120


def test_needle_retry_instruction_picks_read_when_named_in_nl():
    from compositor.tools.maple_nl import needle_retry_instruction

    out = needle_retry_instruction(
        maple_nl='Read path="README.md" for the overview',
        user_task="調べて",
        tool_names=["read", "ls", "grep"],
    )
    assert out.startswith("Call read.")
    assert "README.md" in out


def test_needle_retry_instruction_keeps_create_file_shape():
    from compositor.tools.maple_nl import needle_retry_instruction

    out = needle_retry_instruction(
        maple_nl="please write the file",
        user_task="Create file live.txt with exactly this content:\nok\n",
        tool_names=["write", "read"],
    )
    assert out.startswith("Call write.")
    assert "live.txt" in out
    assert "ok" in out


@pytest.mark.asyncio
async def test_maple_nl_done_keeps_both_content_and_reasoning(tmp_path):
    """Measurement: do not drop the unused Maple channel."""

    class Maple:
        async def chat_completions(self, payload):
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": 'Call ls. path="."',
                            "reasoning_content": "I should list the repo root first.",
                        }
                    }
                ]
            }

    class RT(NeedleToolRuntime):
        def available(self):
            return True

        def complete(self, query, tools, *, max_new_tokens=None):
            return NeedleCompleteResult(
                function_calls=[
                    {
                        "name": "ls",
                        "arguments": {"path": "."},
                    }
                ],
                confidence=0.9,
                reasoning="listing root",
                raw={"function_calls": [{"name": "ls"}], "reasoning": "listing root"},
            )

    c = Compositor(
        Maple(),
        traces=TraceStore(tmp_path),
        tool_path=NeedleToolPath(RT(), prefer_chunk_writes=False),
        needle_via_maple_nl=True,
        maple_tools_primary=False,
        needle_chunk_writes=False,
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
    nl_done = next(e for e in result.trace.events if e.detail.get("phase") == "maple_nl_done")
    assert nl_done.detail["maple_content"] == 'Call ls. path="."'
    assert nl_done.detail["maple_reasoning"] == "I should list the repo root first."
    assert nl_done.detail["maple_nl"] == 'Call ls. path="."'
    needle = next(e for e in result.trace.events if e.detail.get("phase") == "needle_complete")
    assert needle.detail["needle_query"].startswith("Call ls")
    assert needle.detail["needle_function_calls"][0]["name"] == "ls"
    assert needle.detail["needle_reasoning"] == "listing root"
    assert needle.detail["needle_confidence"] == 0.9
    assert needle.detail["attempt"] == "primary"


@pytest.mark.asyncio
async def test_needle_empty_still_logs_complete_trace(tmp_path):
    class Maple:
        async def chat_completions(self, payload):
            if payload.get("tool_choice") == "none":
                return {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": (
                                    "No tool call needed.\n\n"
                                    "Chibi is a bench harness on M1 Max for Qwen throughput."
                                ),
                            }
                        }
                    ]
                }
            raise AssertionError("no fallback")

    class RT(NeedleToolRuntime):
        def available(self):
            return True

        def complete(self, query, tools, *, max_new_tokens=None):
            return NeedleCompleteResult(
                function_calls=[],
                confidence=0.02,
                reasoning="No tool for writing or editing tasks.",
                raw={"function_calls": [], "reasoning": "No tool for writing"},
            )

    c = Compositor(
        Maple(),
        traces=TraceStore(tmp_path),
        tool_path=NeedleToolPath(RT(), prefer_chunk_writes=False),
        needle_via_maple_nl=True,
        maple_tools_primary=False,
        needle_chunk_writes=False,
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
    needle = next(e for e in result.trace.events if e.detail.get("phase") == "needle_complete")
    assert needle.detail["needle_function_calls"] == []
    assert "writing" in (needle.detail.get("needle_reasoning") or "")
    assert any(e.detail.get("phase") == "final_answer_stop" for e in result.trace.events)


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
                                'Use write on Cargo.toml:\n```Cargo.toml\naxum = "0.7"\n```\n'
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
        maple_tools_primary=False,
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
    nl_done = next(e for e in result.trace.events if e.detail.get("phase") == "maple_nl_done")
    assert nl_done.detail["maple_nl"].startswith("Use write on Cargo.toml")
    assert nl_done.detail["maple_nl_source"] == "content"
    assert "axum" in nl_done.detail["maple_nl"]
    ready = next(
        e for e in result.trace.events if e.detail.get("phase") == "needle_instruction_ready"
    )
    assert "Cargo.toml" in ready.detail["needle_instruction"]


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
            q = query.lower()
            if "unusable" in q or "failed tool" in q or "call write" in q or "cargo.toml" in q:
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
        maple_tools_primary=False,
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
    args = result.response["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
    if isinstance(args, str):
        args = json.loads(args)
    assert args["path"] == "Cargo.toml"
    assert "axum" in args["content"]


@pytest.mark.asyncio
async def test_nl_retries_on_degenerate_write(tmp_path):
    """Broken write args must not skip user-task retry (smoke4 hole)."""

    class Maple:
        async def chat_completions(self, payload):
            assert payload.get("tool_choice") == "none"
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Write path=Tower content=0.5",
                        }
                    }
                ]
            }

    class RT(NeedleToolRuntime):
        def __init__(self):
            self.n = 0

        def available(self):
            return True

        def complete(self, query, tools, *, max_new_tokens=None):
            self.n += 1
            if "Call write" in query and "live-nl-retry.txt" in query:
                return NeedleCompleteResult(
                    function_calls=[
                        {
                            "name": "write",
                            "arguments": {
                                "path": "live-nl-retry.txt",
                                "content": "ok-body",
                            },
                        }
                    ],
                    confidence=0.8,
                    reasoning=None,
                    raw={},
                )
            return NeedleCompleteResult(
                function_calls=[
                    {
                        "name": "write",
                        "arguments": {"path": "Tower", "content": "0.5"},
                    }
                ],
                confidence=0.2,
                reasoning=None,
                raw={},
            )

    rt = RT()
    c = Compositor(
        Maple(),
        traces=TraceStore(tmp_path),
        tool_path=NeedleToolPath(rt),
        needle_via_maple_nl=True,
        maple_tools_primary=False,
        needle_chunk_writes=False,
        needle_correct_degenerate=True,
        buffer_side_effects=False,
    )
    result = await c.chat_completions(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Create file live-nl-retry.txt with exactly this content:\n"
                        "ok-body\n\nUse the write tool."
                    ),
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
                },
                {
                    "type": "function",
                    "function": {
                        "name": "read",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                        },
                    },
                },
            ],
        }
    )
    assert rt.n >= 2
    args = result.response["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
    if isinstance(args, str):
        args = json.loads(args)
    assert args["path"] == "live-nl-retry.txt"
    assert args["content"] == "ok-body"
    phases = [
        e.detail.get("phase")
        for e in result.trace.events
        if e.stage == "tool_select" and isinstance(e.detail, dict)
    ]
    assert "needle_quality_miss" in phases
    assert "needle_retry_user_task" in phases


@pytest.mark.asyncio
async def test_no_tool_call_nl_short_circuits_to_stop_without_fallback(tmp_path):
    """Explicit no-tool NL + Needle empty → stop content; no retry/FB."""

    class Maple:
        async def chat_completions(self, payload):
            if payload.get("tool_choice") == "none":
                return {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": (
                                    "No tool call needed.\n\n"
                                    "Chibi is a measurement harness for Qwen3.5-4B "
                                    "on Apple Silicon (M1 Max), targeting 150 tok/s."
                                ),
                            }
                        }
                    ]
                }
            raise AssertionError("fallback Maple must not run")

    class RT(NeedleToolRuntime):
        def __init__(self):
            self.n = 0

        def available(self):
            return True

        def complete(self, query, tools, *, max_new_tokens=None):
            self.n += 1
            return NeedleCompleteResult(
                function_calls=[],
                confidence=0.01,
                reasoning="No tool for writing or editing tasks.",
                raw={},
            )

    rt = RT()
    c = Compositor(
        Maple(),
        traces=TraceStore(tmp_path),
        tool_path=NeedleToolPath(rt, prefer_chunk_writes=False),
        needle_via_maple_nl=True,
        maple_tools_primary=False,
        needle_chunk_writes=False,
        buffer_side_effects=False,
    )
    result = await c.chat_completions(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Explore this repo and explain what it does.",
                }
            ],
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
    assert rt.n == 1
    choice = result.response["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert "Chibi is a measurement harness" in (choice["message"].get("content") or "")
    assert not choice["message"].get("tool_calls")
    phases = [
        e.detail.get("phase")
        for e in result.trace.events
        if e.stage == "tool_select" and isinstance(e.detail, dict)
    ]
    assert "final_answer_stop" in phases
    assert "needle_retry_user_task" not in phases
    assert "fallback_maple" not in phases
    assert result.trace.response_summary.get("needle_via") == "maple_nl_no_tool"


@pytest.mark.asyncio
async def test_nl_empty_call_retry_uses_allowlisted_brief(tmp_path):
    """Abstain → structured retry (not soft user-task) can recover read/ls."""

    class Maple:
        async def chat_completions(self, payload):
            assert payload.get("tool_choice") == "none"
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Call ls on the current directory.",
                        }
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
            if query.startswith("Call ls."):
                return NeedleCompleteResult(
                    function_calls=[{"name": "ls", "arguments": {"path": "."}}],
                    confidence=0.9,
                    reasoning="ok",
                    raw={},
                )
            return NeedleCompleteResult(
                function_calls=[],
                confidence=0.01,
                reasoning="No tool for phone calls",
                raw={},
            )

    rt = RT()
    c = Compositor(
        Maple(),
        traces=TraceStore(tmp_path),
        tool_path=NeedleToolPath(rt),
        needle_via_maple_nl=True,
        maple_tools_primary=False,
        needle_chunk_writes=False,
        buffer_side_effects=False,
    )
    result = await c.chat_completions(
        {
            "messages": [{"role": "user", "content": "リポジトリを調べて"}],
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
                },
                {
                    "type": "function",
                    "function": {
                        "name": "read",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                        },
                    },
                },
            ],
        }
    )
    assert len(rt.queries) >= 2
    assert rt.queries[1].startswith("Call ls.")
    assert "phone" not in rt.queries[1].lower()
    assert "Do not invent" not in rt.queries[1]
    assert result.response["choices"][0]["finish_reason"] == "tool_calls"
    assert result.response["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "ls"
    phases = [
        e.detail.get("phase")
        for e in result.trace.events
        if e.stage == "tool_select" and isinstance(e.detail, dict)
    ]
    assert "needle_retry_user_task" in phases
    assert "fallback_maple" not in phases


@pytest.mark.asyncio
async def test_nl_path_presents_read_without_missing_mutation_coerce(tmp_path):
    """Without MissingMutation, a read-only Maple settle is presented as-is."""

    class Maple:
        async def chat_completions(self, payload):
            if payload.get("tool_choice") == "none":
                return {"choices": [{"message": {"role": "assistant", "content": "hmm"}}]}
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_r",
                                    "type": "function",
                                    "function": {
                                        "name": "read",
                                        "arguments": json.dumps({"path": "notes.txt"}),
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
            return NeedleCompleteResult([], 0.0, "abstain", {})

    c = Compositor(
        Maple(),
        traces=TraceStore(tmp_path),
        tool_path=NeedleToolPath(RT()),
        needle_via_maple_nl=True,
        maple_tools_primary=False,
        needle_chunk_writes=False,
        needle_correct_degenerate=True,
        buffer_side_effects=False,
        repair_attempts=2,
    )
    result = await c.chat_completions(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Implement notes.txt so it contains exactly hello. "
                        "Use the write tool; do not stop after read."
                    ),
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
                },
                {
                    "type": "function",
                    "function": {
                        "name": "read",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                        },
                    },
                },
            ],
        }
    )
    call = result.response["choices"][0]["message"]["tool_calls"][0]
    assert call["function"]["name"] == "read"
    kinds = [
        e.detail.get("kind")
        for e in result.trace.events
        if e.stage == "verify" and isinstance(e.detail, dict)
    ]
    assert "missing_mutation" not in kinds
