"""Deterministic read(directory) → ls coerce."""

from __future__ import annotations

import json
from pathlib import Path

from compositor.tools.coerce_read_ls import coerce_read_directory_to_ls


def _read_call(path: str) -> dict:
    return {
        "id": "call_1",
        "type": "function",
        "function": {
            "name": "read",
            "arguments": json.dumps({"path": path}),
        },
    }


def _response(calls: list[dict]) -> dict:
    return {
        "choices": [
            {
                "message": {"role": "assistant", "content": None, "tool_calls": calls},
                "finish_reason": "tool_calls",
            }
        ]
    }


def test_coerce_dot_path_read_to_ls():
    out = coerce_read_directory_to_ls(_response([_read_call(".")]))
    fn = out["choices"][0]["message"]["tool_calls"][0]["function"]
    assert fn["name"] == "ls"
    assert json.loads(fn["arguments"])["path"] == "."


def test_coerce_trailing_slash_to_ls():
    out = coerce_read_directory_to_ls(_response([_read_call("bench/")]))
    assert out["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "ls"


def test_coerce_absolute_directory_via_fs(tmp_path: Path):
    out = coerce_read_directory_to_ls(_response([_read_call(str(tmp_path))]))
    assert out["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "ls"


def test_does_not_coerce_file_path(tmp_path: Path):
    f = tmp_path / "README.md"
    f.write_text("hi\n", encoding="utf-8")
    out = coerce_read_directory_to_ls(_response([_read_call(str(f))]))
    assert out["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "read"


def test_does_not_coerce_extensionless_file(tmp_path: Path):
    f = tmp_path / "Makefile"
    f.write_text("all:\n\ttrue\n", encoding="utf-8")
    out = coerce_read_directory_to_ls(_response([_read_call(str(f))]))
    assert out["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "read"


def test_leaves_ls_and_write_untouched():
    calls = [
        {
            "id": "c1",
            "type": "function",
            "function": {"name": "ls", "arguments": '{"path":"."}'},
        },
        {
            "id": "c2",
            "type": "function",
            "function": {
                "name": "write",
                "arguments": json.dumps({"path": "a.txt", "content": "x"}),
            },
        },
    ]
    upstream = _response(calls)
    out = coerce_read_directory_to_ls(upstream)
    names = [c["function"]["name"] for c in out["choices"][0]["message"]["tool_calls"]]
    assert names == ["ls", "write"]
