"""Maple ``<tools>`` / ``<tool_call>`` markup prompt + history rewrite."""

from __future__ import annotations

import json

from compositor.tools.maple_tool_markup import (
    maple_tool_markup_request,
    tools_system_suffix,
)
from compositor.tools.pseudo_tool import parse_pseudo_tool_calls


def _ls_read_tools() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "ls",
                "description": "list",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read",
                "description": "read",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        },
    ]


def test_tools_system_suffix_wraps_tools_xml():
    text = tools_system_suffix(_ls_read_tools())
    assert "<tools>" in text and "</tools>" in text
    assert "<tool_call>" in text
    assert '"name": "ls"' in text or '"name":"ls"' in text or "ls" in text
    assert "plain text" in text.lower()


def test_maple_tool_markup_request_pops_tools_and_injects_system():
    req = {
        "messages": [{"role": "user", "content": "list root"}],
        "tools": _ls_read_tools(),
        "tool_choice": "auto",
    }
    out = maple_tool_markup_request(req)
    assert out is not req
    assert "tools" not in out
    assert "tool_choice" not in out
    assert out["messages"][0]["role"] == "system"
    assert "<tools>" in out["messages"][0]["content"]
    assert out["messages"][-1]["content"] == "list root"


def test_maple_tool_markup_rewrites_history_tool_calls_and_results():
    req = {
        "messages": [
            {"role": "user", "content": "read it"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {
                            "name": "read",
                            "arguments": '{"path": "README.md"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "c1",
                "content": "# evprtr\n",
            },
        ],
        "tools": _ls_read_tools(),
    }
    out = maple_tool_markup_request(req)
    roles = [m["role"] for m in out["messages"]]
    assert roles[0] == "system"
    assert "tool" not in roles
    assistant = next(m for m in out["messages"] if m["role"] == "assistant")
    assert "<tool_call>" in assistant["content"]
    assert "read" in assistant["content"]
    assert "tool_calls" not in assistant
    tool_user = [m for m in out["messages"] if m["role"] == "user"][-1]
    assert "<tool_response>" in tool_user["content"]
    assert "# evprtr" in tool_user["content"]


def test_parse_hf_style_arguments_key():
    text = (
        '<tool_call>\n'
        '{"name": "ls", "arguments": {"path": ".", "limit": 50}}\n'
        "</tool_call>"
    )
    calls = parse_pseudo_tool_calls(text)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "ls"
    assert json.loads(calls[0]["function"]["arguments"]) == {
        "path": ".",
        "limit": 50,
    }


def test_markup_skips_when_tool_choice_none():
    req = {
        "messages": [{"role": "user", "content": "hi"}],
        "tools": _ls_read_tools(),
        "tool_choice": "none",
    }
    assert maple_tool_markup_request(req) is req
