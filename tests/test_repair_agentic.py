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
