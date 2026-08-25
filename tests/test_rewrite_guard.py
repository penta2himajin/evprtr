from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness" / "pi"))

from rewrite_guard import (  # noqa: E402
    RewriteGuard,
    fingerprint_from_confirm_message,
    fingerprint_from_input,
    fingerprint_write,
)


def test_fingerprint_write_stable():
    a = fingerprint_write("src/main.rs", "hello")
    b = fingerprint_write("src/main.rs", "hello")
    c = fingerprint_write("src/main.rs", "hello!")
    assert a == b
    assert a != c
    assert a.startswith("write:src/main.rs:")


def test_fingerprint_from_input_edit():
    fp = fingerprint_from_input(
        "edit",
        {"path": "a.rs", "edits": [{"oldText": "x", "newText": "y"}]},
    )
    assert fp.startswith("edit:a.rs:")


def test_fingerprint_from_confirm_prefers_fp_line():
    msg = (
        "fp=write:src/main.rs:deadbeefdeadbeefdeadbeefdeadbeef\n"
        "[appr-1]\n"
        "path=src/main.rs content_len=10\n---\nhello"
    )
    assert (
        fingerprint_from_confirm_message(msg)
        == "write:src/main.rs:deadbeefdeadbeefdeadbeefdeadbeef"
    )


def test_guard_auto_rejects_identical_pending():
    g = RewriteGuard(identical_pending_limit=2, identical_pending_abort_at=3)
    assert g.on_pending("fp1") == ("wait", "")
    action, note = g.on_pending("fp1")
    assert action == "auto_reject"
    assert "repeated" in note
    action, note = g.on_pending("fp1")
    assert action == "abort"


def test_guard_aborts_rewrite_after_approve():
    g = RewriteGuard()
    assert g.on_pending("fpW")[0] == "wait"
    g.on_approved("fpW")
    action, note = g.on_pending("fpW")
    assert action == "abort"
    assert "already-approved" in note


def test_guard_different_fps_wait():
    g = RewriteGuard()
    assert g.on_pending("a")[0] == "wait"
    assert g.on_pending("b")[0] == "wait"
    assert g.on_pending("b")[0] == "auto_reject"
