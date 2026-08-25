from __future__ import annotations

from compositor.verify.repair import MaplePreviewRepair, request_has_tools
from compositor.verify.types import Diagnosis


def test_request_has_tools():
    assert request_has_tools({"tools": [{"type": "function", "function": {"name": "write"}}]})
    assert not request_has_tools({"tools": []})
    assert not request_has_tools(
        {"tools": [{"type": "function"}], "tool_choice": "none"}
    )


def test_maple_repair_with_tools_stays_agentic():
    repair = MaplePreviewRepair()
    request = {
        "messages": [{"role": "user", "content": "Add axum to Cargo.toml via write tool"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "write",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        "max_tokens": 2048,
    }
    diagnosis = Diagnosis(
        policy_id="generic.repetition",
        field="reasoning_content",
        kind="char_motif",
        onset=10,
        detail={},
    )
    payload = repair.build_payload(request, attempt=1, diagnosis=diagnosis)
    assert payload["tools"]
    assert payload.get("tool_choice") != "none"
    assert payload["max_tokens"] >= 768
    system = payload["messages"][0]["content"]
    user = payload["messages"][1]["content"]
    assert "coding agent" in system.lower()
    assert "tool call" in user.lower()
    assert "Emit ONLY" not in user


def test_original_user_text_prefers_primary_task():
    from compositor.verify.repair import original_user_text

    request = {
        "messages": [
            {"role": "user", "content": "Implement HTTP server per plan: add axum, tests, bind 3000."},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "agent"},
            {"role": "user", "content": "continue"},
        ]
    }
    text = original_user_text(request)
    assert "Implement HTTP server" in text
    assert text != "agent"


def test_pseudo_markup_repair_keeps_tools_when_present():
    repair = MaplePreviewRepair()
    request = {
        "messages": [
            {
                "role": "user",
                "content": "Write Cargo.toml with axum using the write tool only.",
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {"name": "write", "parameters": {"type": "object"}},
            }
        ],
    }
    diagnosis = Diagnosis(
        policy_id="maple_preview.pseudo_tool_markup",
        field="content",
        kind="pseudo_tool_markup",
        onset=0,
        detail={},
    )
    payload = repair.build_payload(request, attempt=1, diagnosis=diagnosis)
    assert payload.get("tool_choice") != "none"
    assert payload["tools"]
    assert "structured tool call" in payload["messages"][1]["content"].lower()
