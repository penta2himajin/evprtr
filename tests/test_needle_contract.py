"""Needle 2 contract: short queries, ≤5 tools, confidence / token defaults."""

from __future__ import annotations

from compositor.tools.needle_contract import (
    needle_system_facts,
    normalize_needle_query,
    select_needle_tools,
)


def _many_tools() -> list[dict]:
    names = [
        "write",
        "edit",
        "read",
        "ls",
        "grep",
        "find",
        "bash",
        "web_search",
        "browser",
    ]
    return [
        {
            "name": n,
            "description": f"{n} tool",
            "parameters": {"type": "object", "properties": {}},
        }
        for n in names
    ]


def test_select_needle_tools_passthrough_when_few():
    tools = _many_tools()[:3]
    assert select_needle_tools(tools) == tools


def test_select_needle_tools_caps_at_five_and_prefers_hint():
    tools = _many_tools()
    picked = select_needle_tools(tools, hint='Call bash. command="wc -l hello.txt"', max_tools=5)
    assert len(picked) == 5
    names = [t["name"] for t in picked]
    assert "bash" in names
    # Must not silently keep all nine.
    assert "web_search" not in names or "browser" not in names


def test_select_needle_tools_priority_when_no_hint():
    tools = _many_tools()
    picked = select_needle_tools(tools, hint="", max_tools=5)
    names = [t["name"] for t in picked]
    assert names == ["write", "edit", "read", "ls", "grep"]


def test_normalize_needle_query_keeps_short_call():
    out = normalize_needle_query(
        'Call ls. path="."',
        user_task="list files",
        tool_names=["ls", "read"],
    )
    assert out.startswith("Call ls.")


def test_normalize_needle_query_shortens_essay():
    essay = (
        "First I should carefully consider the repository structure and then "
        "perhaps list the files in the current directory using the ls tool "
        "with an appropriate path argument before reading anything else.\n" * 5
    )
    out = normalize_needle_query(
        essay,
        user_task="List files in the repo root.",
        tool_names=["ls", "read", "grep"],
        limit=160,
    )
    assert len(out) <= 160
    assert out.lower().startswith("call ")


def test_needle_system_facts_shape():
    facts = needle_system_facts(cwd="/tmp/proj")
    assert "date:" in facts
    assert "locale: en-US" in facts
    assert "device: coding-agent" in facts
    assert "location: /tmp/proj" in facts
