"""Synthesize oMLX structured_outputs from OpenAI tools; promote JSON content."""

from __future__ import annotations

import json

from compositor.tools.tools_grammar import (
    attach_tools_structured_outputs,
    json_schema_from_openai_tools,
    promote_tool_json_content,
)


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


def test_json_schema_one_of_tool_names():
    schema = json_schema_from_openai_tools(_ls_read_tools())
    assert "oneOf" in schema
    names = []
    message_branch = False
    for branch in schema["oneOf"]:
        props = branch.get("properties") or {}
        if "message" in props and "name" not in props:
            message_branch = True
            continue
        name_schema = props["name"]
        if "const" in name_schema:
            names.append(name_schema["const"])
        else:
            names.extend(name_schema.get("enum") or [])
    assert set(names) == {"ls", "read"}
    assert message_branch
    ls_branch = next(b for b in schema["oneOf"] if b.get("properties", {}).get("name", {}).get("const") == "ls")
    assert ls_branch["properties"]["arguments"]["required"] == ["path"]


def test_promote_message_json_to_assistant_content():
    upstream = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": json.dumps(
                        {"message": "# evprtr\nentries: 12"},
                        ensure_ascii=False,
                    ),
                },
                "finish_reason": "stop",
            }
        ]
    }
    out = promote_tool_json_content(upstream)
    msg = out["choices"][0]["message"]
    assert msg.get("tool_calls") in (None, [])
    assert msg["content"] == "# evprtr\nentries: 12"
    assert out["choices"][0]["finish_reason"] == "stop"
def test_attach_injects_when_missing_and_tools_present():
    req = {
        "messages": [{"role": "user", "content": "ls ."}],
        "tools": _ls_read_tools(),
        "tool_choice": "auto",
    }
    out = attach_tools_structured_outputs(req)
    assert out is not req
    assert "structured_outputs" in out
    assert "json" in out["structured_outputs"]
    assert out["tools"] == req["tools"]


def test_attach_preserves_harness_structured_outputs():
    req = {
        "messages": [{"role": "user", "content": "hi"}],
        "tools": _ls_read_tools(),
        "structured_outputs": {"grammar": 'root ::= "ZQX"'},
    }
    out = attach_tools_structured_outputs(req)
    assert out["structured_outputs"] == {"grammar": 'root ::= "ZQX"'}


def test_attach_skips_when_tool_choice_none():
    req = {
        "tools": _ls_read_tools(),
        "tool_choice": "none",
        "messages": [],
    }
    out = attach_tools_structured_outputs(req)
    assert "structured_outputs" not in out


def test_promote_tool_json_content_to_tool_calls():
    upstream = {
        "id": "x",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": json.dumps(
                        {"name": "ls", "arguments": {"path": "."}},
                        ensure_ascii=False,
                    ),
                },
                "finish_reason": "stop",
            }
        ],
    }
    out = promote_tool_json_content(upstream)
    calls = out["choices"][0]["message"]["tool_calls"]
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "ls"
    assert json.loads(calls[0]["function"]["arguments"]) == {"path": "."}
    assert out["choices"][0]["finish_reason"] == "tool_calls"
    assert out["choices"][0]["message"].get("content") in (None, "")


def test_promote_leaves_prose_alone():
    upstream = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "No tool call needed. Here is the answer.",
                },
                "finish_reason": "stop",
            }
        ]
    }
    out = promote_tool_json_content(upstream)
    assert out["choices"][0]["message"].get("tool_calls") in (None, [])
    assert "No tool call needed" in out["choices"][0]["message"]["content"]
