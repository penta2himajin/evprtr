"""Deterministic read(directory) ↔ ls(file) coerce."""

from __future__ import annotations

import json
from pathlib import Path

from compositor.tools.coerce_read_ls import (
    coerce_ls_file_to_read,
    coerce_read_directory_to_ls,
)


def _read_call(path: str) -> dict:
    return {
        "id": "call_1",
        "type": "function",
        "function": {
            "name": "read",
            "arguments": json.dumps({"path": path}),
        },
    }


def _ls_call(path: str, call_id: str = "call_1") -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": "ls",
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


def test_coerce_ls_markdown_path_to_read():
    out = coerce_ls_file_to_read(_response([_ls_call("README.md")]))
    fn = out["choices"][0]["message"]["tool_calls"][0]["function"]
    assert fn["name"] == "read"
    assert json.loads(fn["arguments"])["path"] == "README.md"


def test_coerce_ls_absolute_file_via_fs(tmp_path: Path):
    f = tmp_path / "notes.py"
    f.write_text("x = 1\n", encoding="utf-8")
    out = coerce_ls_file_to_read(_response([_ls_call(str(f))]))
    assert out["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "read"


def test_does_not_coerce_ls_dot():
    out = coerce_ls_file_to_read(_response([_ls_call(".")]))
    assert out["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "ls"


def test_does_not_coerce_ls_directory(tmp_path: Path):
    out = coerce_ls_file_to_read(_response([_ls_call(str(tmp_path))]))
    assert out["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "ls"


def test_does_not_coerce_ls_trailing_slash():
    out = coerce_ls_file_to_read(_response([_ls_call("bench/")]))
    assert out["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "ls"


def test_leaves_read_and_write_untouched_for_ls_coerce():
    calls = [
        _read_call("README.md"),
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
    out = coerce_ls_file_to_read(upstream)
    names = [c["function"]["name"] for c in out["choices"][0]["message"]["tool_calls"]]
    assert names == ["read", "write"]
