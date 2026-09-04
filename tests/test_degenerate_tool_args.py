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


def test_empty_edit_texts_detected():
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
                            {"path": "notes.txt", "edits": [{}]}
                        ),
                    },
                }
            ],
        }
    )
    assert hit is not None
    assert hit.detail["reason"] == "empty_edit_texts"


def test_align_create_file_rewrites_mangled_path():
    from compositor.tools.maple_nl import align_create_file_tool_calls

    task = (
        "Create file live-nl-smoke5.txt with exactly this content:\n"
        "nl-needle-ok\n\nUse the write tool."
    )
    upstream = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        _write_call("file_live-nl-smoke5.txt", "nl-needle-ok")
                    ],
                }
            }
        ]
    }
    out = align_create_file_tool_calls(upstream, task)
    args = json.loads(out["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])
    assert args["path"] == "live-nl-smoke5.txt"
    assert args["content"] == "nl-needle-ok"


def test_parse_create_file_stops_at_harness_blurb():
    from compositor.tools.maple_nl import parse_create_file

    task = (
        "Create file live-nl-smoke6b.txt with exactly this content:\n"
        "second-ok\n\n"
        "You may read first if needed, but you must end by calling write "
        "with the file. Do not stop after read."
    )
    parsed = parse_create_file(task)
    assert parsed is not None
    path, body = parsed
    assert path == "live-nl-smoke6b.txt"
    assert body == "second-ok"


def test_synthetic_create_and_edit():
    from compositor.tools.maple_nl import (
        synthetic_create_file_response,
        synthetic_edit_replace_response,
        synthetic_mutation_response,
    )

    w = synthetic_create_file_response(
        "Create file a.txt with exactly this content:\nhi\n\nUse the write tool."
    )
    assert w is not None
    args = json.loads(w["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])
    assert args == {"path": "a.txt", "content": "hi"}

    e = synthetic_edit_replace_response(
        'Edit the file C:\\tmp\\x.txt: replace oldText "alpha" with newText "gamma".'
    )
    assert e is not None
    eargs = json.loads(e["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])
    assert eargs["path"].endswith("x.txt")
    assert eargs["edits"][0] == {"oldText": "alpha", "newText": "gamma"}

    # Public forge entry is disabled (helpers remain until deleted).
    assert synthetic_mutation_response(
        "Create file a.txt with exactly this content:\nhi\n"
    ) is None


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


def test_missing_mutation_detector_not_in_default_bundle():
    """Keyword mutation classification retired from the default verify stack."""
    from compositor.verify.detectors_tools import MISSING_MUTATION_ID, MissingMutationDetector

    bundle = VerifyBundle.maple_preview()
    assert all(d.policy_id != MISSING_MUTATION_ID for d in bundle.detectors)
    assert MissingMutationDetector.policy_id == MISSING_MUTATION_ID


def test_bundle_does_not_flag_read_only_on_create_task():
    bundle = VerifyBundle.maple_preview()
    request = {
        "messages": [
            {
                "role": "user",
                "content": "Create file notes.txt with content hello via write tool",
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {"name": "write", "parameters": {"type": "object"}},
            },
            {
                "type": "function",
                "function": {"name": "read", "parameters": {"type": "object"}},
            },
        ],
    }
    upstream = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {
                                "name": "read",
                                "arguments": '{"path": "notes.txt"}',
                            },
                        }
                    ],
                }
            }
        ]
    }
    assert bundle.diagnose(upstream, request=request) is None
