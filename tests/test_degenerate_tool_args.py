from __future__ import annotations

import json

from compositor.verify.detectors_tools import DegenerateToolCallDetector
from compositor.verify.pipeline import VerifyBundle
from compositor.verify.repair import MaplePreviewRepair
from compositor.verify.types import Diagnosis


def _write_call(path: str, content: str) -> dict:
    return {
        "id": "call_x",
        "type": "function",
        "function": {
            "name": "write",
            "arguments": json.dumps({"path": path, "content": content}),
        },
    }


def test_degenerate_write_path_not_filelike():
    d = DegenerateToolCallDetector()
    hit = d.diagnose(
        {"role": "assistant", "content": None, "tool_calls": [_write_call("Tower", "0.5")]}
    )
    assert hit is not None
    assert hit.kind == "degenerate_tool_args"
    assert hit.detail["reason"] == "path_not_filelike"


def test_degenerate_write_cargo_too_thin():
    d = DegenerateToolCallDetector()
    hit = d.diagnose(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [_write_call("Cargo.toml", "Cargo.toml")],
        }
    )
    assert hit is not None
    assert hit.detail["reason"] in {"content_equals_path", "cargo_toml_too_thin"}


def test_valid_write_passes():
    d = DegenerateToolCallDetector()
    body = '[package]\nname = "tinyserve"\n\n[dependencies]\naxum = "0.7"\n'
    hit = d.diagnose(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [_write_call("Cargo.toml", body)],
        }
    )
    assert hit is None


def test_noop_edit_detected():
    d = DegenerateToolCallDetector()
    hit = d.diagnose(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_y",
                    "type": "function",
                    "function": {
                        "name": "edit",
                        "arguments": json.dumps(
                            {
                                "path": "Cargo.toml",
                                "edits": [{"oldText": "MIT", "newText": "MIT"}],
                            }
                        ),
                    },
                }
            ],
        }
    )
    assert hit is not None
    assert hit.detail["reason"] == "noop_edit"


def test_repair_degenerate_stays_agentic():
    repair = MaplePreviewRepair()
    request = {
        "messages": [{"role": "user", "content": "Write Cargo.toml with axum deps."}],
        "tools": [{"type": "function", "function": {"name": "write"}}],
    }
    diagnosis = Diagnosis(
        policy_id="maple_preview.degenerate_tool_args",
        field="tool_calls",
        kind="degenerate_tool_args",
        detail={"tool": "write", "reason": "cargo_toml_too_thin", "path": "Cargo.toml"},
    )
    payload = repair.build_payload(request, attempt=1, diagnosis=diagnosis)
    assert payload.get("tool_choice") != "none"
    assert "unusable arguments" in payload["messages"][1]["content"].lower()


def test_bundle_sanitizes_degenerate_tool_calls():
    bundle = VerifyBundle.maple_preview()
    upstream = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [_write_call("Tower", "0.5")],
                }
            }
        ]
    }
    diagnosis = bundle.diagnose(upstream)
    assert diagnosis is not None
    assert diagnosis.kind == "degenerate_tool_args"
    cleaned = bundle.sanitize(upstream, diagnosis)
    msg = cleaned["choices"][0]["message"]
    assert msg.get("tool_calls") in (None, [])
