#!/usr/bin/env python3
"""Pi RPC bridge for path-B side-effect approvals.

Spawns ``pi --mode rpc`` with the evprtr gate extension. When the gate calls
``ctx.ui.confirm``, Pi emits ``extension_ui_request``; this bridge parks the
request under ``state-dir/pending/`` until a supervisor runs ``decide``.

Typical Cursor / supervisor loop::

    # terminal A (or background)
    python harness/pi/rpc_bridge.py run --cwd C:/repos/tinyserve --prompt "..."

    # terminal B / next agent turn
    python harness/pi/rpc_bridge.py pending --state-dir ...
    python harness/pi/rpc_bridge.py decide --id <uuid> --approve --state-dir ...
    python harness/pi/rpc_bridge.py wait --state-dir ...

JSONL framing uses LF only (Pi RPC requirement). Do not use readline.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# Allow `python harness/pi/rpc_bridge.py` without installing the package.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from rewrite_guard import RewriteGuard, fingerprint_from_confirm_message  # noqa: E402


def _default_ext() -> Path:
    return Path(__file__).resolve().parent / "evprtr-side-effect-gate.ts"


def _state_dir(cwd: Path, explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).resolve()
    return (cwd / ".evprtr" / "pi-rpc").resolve()


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _append_event(state: Path, event: dict[str, Any]) -> None:
    log = state / "events.jsonl"
    state.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(event, ensure_ascii=False) + "\n")


def _set_status(state: Path, **fields: Any) -> None:
    path = state / "status.json"
    cur: dict[str, Any] = {}
    if path.is_file():
        try:
            cur = _read_json(path)
        except json.JSONDecodeError:
            cur = {}
    cur.update(fields)
    cur["updated_at"] = time.time()
    _write_json(path, cur)


def _send(proc: subprocess.Popen[bytes], obj: dict[str, Any]) -> None:
    assert proc.stdin is not None
    line = (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")
    proc.stdin.write(line)
    proc.stdin.flush()


def _wait_decision(state: Path, req_id: str, timeout: float | None) -> dict[str, Any]:
    pending = state / "pending" / f"{req_id}.json"
    decision = state / "decisions" / f"{req_id}.json"
    deadline = None if timeout is None else time.time() + timeout
    while True:
        if decision.is_file():
            data = _read_json(decision)
            try:
                decision.unlink()
            except OSError:
                pass
            if pending.is_file():
                try:
                    pending.unlink()
                except OSError:
                    pass
            return data
        if deadline is not None and time.time() >= deadline:
            return {"cancelled": True, "reason": "decision timeout"}
        time.sleep(0.2)


def cmd_run(args: argparse.Namespace) -> int:
    cwd = Path(args.cwd).resolve()
    state = _state_dir(cwd, args.state_dir)
    if state.exists() and args.fresh_state:
        shutil.rmtree(state)
    (state / "pending").mkdir(parents=True, exist_ok=True)
    (state / "decisions").mkdir(parents=True, exist_ok=True)
    (state / "events.jsonl").write_text("", encoding="utf-8")

    prompt = args.prompt
    if args.prompt_file:
        prompt = Path(args.prompt_file).resolve().read_text(encoding="utf-8")
    if not prompt or not str(prompt).strip():
        print("provide --prompt or --prompt-file", file=sys.stderr)
        return 2
    args.prompt = str(prompt)

    ext = Path(args.extension).resolve() if args.extension else _default_ext()
    if not ext.is_file():
        print(f"extension missing: {ext}", file=sys.stderr)
        return 2

    pi_bin = shutil.which("pi") or shutil.which("pi.cmd")
    if not pi_bin:
        print("pi not found on PATH", file=sys.stderr)
        return 2

    cmd = [
        pi_bin,
        "--mode",
        "rpc",
        "--provider",
        args.provider,
        "--model",
        args.model,
        "--offline",
        "--approve",
        "-e",
        str(ext),
        "--session-dir",
        str(Path(args.session_dir).resolve())
        if args.session_dir
        else str(state / "sessions"),
    ]
    if args.name:
        cmd.extend(["--name", args.name])
    if args.no_session:
        cmd.append("--no-session")
    if args.tools:
        cmd.extend(["--tools", args.tools])
    if args.exclude_tools:
        cmd.extend(["--exclude-tools", args.exclude_tools])

    env = os.environ.copy()
    if args.evprtr_base_url:
        env["EVPRTR_BASE_URL"] = args.evprtr_base_url
    # Never inherit auto-approve; path B requires explicit RPC decide.
    env.pop("EVPRTR_PI_AUTO_APPROVE", None)

    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        bufsize=0,
    )
    _write_json(state / "bridge.json", {"pid": proc.pid, "cwd": str(cwd), "cmd": cmd})
    _set_status(state, phase="starting", pid=proc.pid, exit_code=None)

    assert proc.stdout is not None
    buf = b""
    settled = False
    prompt_sent = False
    reject_count = 0
    guard = RewriteGuard(
        identical_pending_limit=max(1, int(args.identical_pending_reject_at)),
        identical_pending_abort_at=max(
            max(1, int(args.identical_pending_reject_at)) + 1,
            int(args.identical_pending_abort_at),
        ),
    )

    def _abort_run(phase: str, **extra: Any) -> None:
        nonlocal settled
        print(f"RPC_BRIDGE: abort ({phase}) {extra}", flush=True)
        _set_status(state, phase=phase, reject_count=reject_count, **extra)
        try:
            _send(proc, {"type": "abort"})
        except Exception:
            pass
        settled = False

    def _apply_confirm(req_id: str, confirmed: bool, note: str) -> None:
        nonlocal reject_count
        _send(
            proc,
            {
                "type": "extension_ui_response",
                "id": req_id,
                "confirmed": confirmed,
            },
        )
        if confirmed:
            reject_count = 0
        else:
            reject_count += 1
        _append_event(
            state,
            {
                "type": "bridge_decision",
                "id": req_id,
                "confirmed": confirmed,
                "note": note,
                "auto": True,
            },
        )
        print(
            f"RPC_BRIDGE: auto_decision id={req_id} confirmed={confirmed} note={note!r}",
            flush=True,
        )

    try:
        # Give RPC a moment to boot, then send the prompt.
        time.sleep(0.8)
        if proc.poll() is not None:
            err = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
            print(f"pi exited early: {err}", file=sys.stderr)
            _set_status(state, phase="dead", exit_code=proc.returncode)
            return proc.returncode or 1

        _send(proc, {"id": "prompt-1", "type": "prompt", "message": args.prompt})
        prompt_sent = True
        _set_status(state, phase="running", prompt_sent=True)
        print(f"RPC_BRIDGE: prompt sent; state={state}", flush=True)

        while True:
            chunk = proc.stdout.read(4096)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                raw, buf = buf.split(b"\n", 1)
                if raw.endswith(b"\r"):
                    raw = raw[:-1]
                if not raw:
                    continue
                try:
                    event = json.loads(raw.decode("utf-8"))
                except json.JSONDecodeError:
                    _append_event(
                        state,
                        {
                            "type": "parse_error",
                            "raw": raw.decode("utf-8", "replace"),
                        },
                    )
                    continue

                _append_event(state, event)
                et = event.get("type")

                if et == "extension_ui_request":
                    method = event.get("method")
                    req_id = str(event.get("id"))
                    ui_noise = {
                        "notify",
                        "setStatus",
                        "setWidget",
                        "setTitle",
                        "set_editor_text",
                    }
                    if method in ui_noise:
                        tip = event.get("message") or event.get("title") or ""
                        print(f"RPC_BRIDGE: ui {method}: {tip}", flush=True)
                        continue

                    pending = {
                        "id": req_id,
                        "method": method,
                        "title": event.get("title"),
                        "message": event.get("message"),
                        "options": event.get("options"),
                        "received_at": time.time(),
                    }
                    _write_json(state / "pending" / f"{req_id}.json", pending)
                    _set_status(state, phase="awaiting_decision", pending_id=req_id)
                    print(
                        "RPC_BRIDGE: PENDING_APPROVAL "
                        f"id={req_id} method={method} title={pending.get('title')!r} "
                        f"message={pending.get('message')!r}",
                        flush=True,
                    )

                    fp = (
                        fingerprint_from_confirm_message(str(pending.get("message") or ""))
                        if method == "confirm"
                        else None
                    )
                    if method == "confirm" and fp:
                        action, note = guard.on_pending(fp)
                        if action in {"auto_reject", "abort"}:
                            _apply_confirm(req_id, False, note)
                            # Clear pending file so status/pending stay consistent.
                            try:
                                (state / "pending" / f"{req_id}.json").unlink()
                            except OSError:
                                pass
                            _set_status(state, phase="running", pending_id=None)
                            if action == "abort" or (
                                args.max_rejects and reject_count >= args.max_rejects
                            ):
                                _abort_run(
                                    "aborted_rewrite_loop"
                                    if action == "abort"
                                    else "aborted_rejects",
                                    fingerprint=fp,
                                    note=note,
                                )
                                break
                            continue

                    decision = _wait_decision(state, req_id, args.decision_timeout)
                    if decision.get("cancelled"):
                        _send(
                            proc,
                            {
                                "type": "extension_ui_response",
                                "id": req_id,
                                "cancelled": True,
                            },
                        )
                    elif method == "confirm":
                        confirmed = bool(decision.get("confirmed"))
                        _send(
                            proc,
                            {
                                "type": "extension_ui_response",
                                "id": req_id,
                                "confirmed": confirmed,
                            },
                        )
                        if confirmed:
                            guard.on_approved(fp)
                            reject_count = 0
                        else:
                            guard.on_rejected(fp)
                            reject_count += 1
                            if args.max_rejects and reject_count >= args.max_rejects:
                                _abort_run(
                                    "aborted_rejects",
                                    fingerprint=fp,
                                )
                                break
                    elif method == "select":
                        _send(
                            proc,
                            {
                                "type": "extension_ui_response",
                                "id": req_id,
                                "value": decision.get("value"),
                            },
                        )
                    else:
                        _send(
                            proc,
                            {
                                "type": "extension_ui_response",
                                "id": req_id,
                                "cancelled": True,
                            },
                        )
                    _set_status(state, phase="running", pending_id=None)
                    dumped = json.dumps(decision, ensure_ascii=False)
                    print(
                        f"RPC_BRIDGE: decision applied id={req_id} {dumped}",
                        flush=True,
                    )
                    if state.joinpath("status.json").is_file():
                        st = _read_json(state / "status.json")
                        if st.get("phase") in {
                            "aborted_rejects",
                            "aborted_rewrite_loop",
                        }:
                            break
                    continue

                if et == "tool_execution_start":
                    print(f"RPC_BRIDGE: tool_start {event.get('toolName')}", flush=True)
                elif et == "agent_settled":
                    settled = True
                    _set_status(state, phase="settled")
                    print("RPC_BRIDGE: agent_settled", flush=True)
                elif et == "response" and event.get("success") is False:
                    print(
                        f"RPC_BRIDGE: command error {event.get('command')}: {event.get('error')}",
                        flush=True,
                    )

            if settled and proc.poll() is not None:
                break
            if settled:
                # Agent finished turn; exit cleanly after brief drain.
                time.sleep(0.3)
                break
            st_path = state / "status.json"
            if st_path.is_file():
                try:
                    if _read_json(st_path).get("phase") in {
                        "aborted_rejects",
                        "aborted_rewrite_loop",
                    }:
                        break
                except json.JSONDecodeError:
                    pass

    finally:
        if proc.poll() is None:
            try:
                _send(proc, {"type": "abort"})
            except Exception:
                pass
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        err = ""
        if proc.stderr:
            err = proc.stderr.read().decode("utf-8", errors="replace")
        if err.strip():
            (state / "stderr.txt").write_text(err, encoding="utf-8")
        _set_status(
            state,
            phase="exited",
            exit_code=proc.returncode,
            settled=settled,
            prompt_sent=prompt_sent,
        )

    return 0 if settled else (proc.returncode or 1)


def cmd_pending(args: argparse.Namespace) -> int:
    state = Path(args.state_dir).resolve()
    pending_dir = state / "pending"
    items = []
    if pending_dir.is_dir():
        for path in sorted(pending_dir.glob("*.json")):
            items.append(_read_json(path))
    print(json.dumps({"object": "list", "data": items}, ensure_ascii=False, indent=2))
    return 0


def cmd_decide(args: argparse.Namespace) -> int:
    state = Path(args.state_dir).resolve()
    req_id = args.id
    pending = state / "pending" / f"{req_id}.json"
    if not pending.is_file():
        print(f"no pending request: {req_id}", file=sys.stderr)
        return 1
    if args.approve:
        payload = {"confirmed": True, "value": "Yes", "note": args.note}
    elif args.reject:
        payload = {"confirmed": False, "value": "No", "note": args.note, "cancelled": False}
    else:
        payload = {"cancelled": True, "note": args.note}
    _write_json(state / "decisions" / f"{req_id}.json", payload)
    print(json.dumps({"ok": True, "id": req_id, "decision": payload}, ensure_ascii=False))
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    state = Path(args.state_dir).resolve()
    path = state / "status.json"
    if not path.is_file():
        print(json.dumps({"phase": "missing", "state_dir": str(state)}))
        return 1
    print(json.dumps(_read_json(path), ensure_ascii=False, indent=2))
    return 0


def cmd_wait(args: argparse.Namespace) -> int:
    state = Path(args.state_dir).resolve()
    deadline = time.time() + args.timeout
    while time.time() < deadline:
        path = state / "status.json"
        if path.is_file():
            st = _read_json(path)
            phase = st.get("phase")
            if phase in {"settled", "exited"}:
                print(json.dumps(st, ensure_ascii=False, indent=2))
                return 0 if st.get("settled") or st.get("phase") == "settled" else 1
            if phase == "awaiting_decision":
                print(json.dumps(st, ensure_ascii=False, indent=2))
                return 2
        time.sleep(0.5)
    print(json.dumps({"phase": "timeout", "state_dir": str(state)}), file=sys.stderr)
    return 3


def main(argv: list[str] | None = None) -> int:
    # Windows consoles (cp932) choke on em dashes in tool previews; never let
    # logging encode errors stall a pending decision wait.
    try:
        sys.stdout.reconfigure(errors="replace")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    parser = argparse.ArgumentParser(description="evprtr Pi RPC approval bridge")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Spawn pi --mode rpc and serve approval stalls")
    run.add_argument("--cwd", required=True)
    run.add_argument("--prompt")
    run.add_argument(
        "--prompt-file",
        help="Read prompt text from a UTF-8 file (avoids shell quoting issues)",
    )
    run.add_argument("--state-dir")
    run.add_argument("--extension")
    run.add_argument("--provider", default="evprtr")
    run.add_argument("--model", default="evprtr")
    run.add_argument("--session-dir")
    run.add_argument("--name")
    run.add_argument("--no-session", action="store_true")
    run.add_argument("--fresh-state", action="store_true")
    run.add_argument(
        "--tools",
        help="Pass through to pi --tools (comma-separated allowlist)",
    )
    run.add_argument(
        "--exclude-tools",
        help="Pass through to pi --exclude-tools",
    )
    run.add_argument("--evprtr-base-url", default="http://127.0.0.1:8741")
    run.add_argument(
        "--decision-timeout",
        type=float,
        default=None,
        help="Seconds to wait for each decide (default: forever)",
    )
    run.add_argument(
        "--max-rejects",
        type=int,
        default=12,
        help="Abort after this many rejected gated confirms in one run (0=never)",
    )
    run.add_argument(
        "--identical-pending-reject-at",
        type=int,
        default=2,
        help="Auto-reject when the same gated confirm fingerprint repeats this many times",
    )
    run.add_argument(
        "--identical-pending-abort-at",
        type=int,
        default=3,
        help="Abort when the same gated confirm fingerprint repeats this many times",
    )
    run.set_defaults(func=cmd_run)

    pending = sub.add_parser("pending", help="List pending UI approvals")
    pending.add_argument("--state-dir", required=True)
    pending.set_defaults(func=cmd_pending)

    decide = sub.add_parser("decide", help="Approve/reject a pending confirm")
    decide.add_argument("--state-dir", required=True)
    decide.add_argument("--id", required=True)
    g = decide.add_mutually_exclusive_group(required=True)
    g.add_argument("--approve", action="store_true")
    g.add_argument("--reject", action="store_true")
    g.add_argument("--cancel", action="store_true")
    decide.add_argument("--note")
    decide.set_defaults(func=cmd_decide)

    status = sub.add_parser("status", help="Show bridge status.json")
    status.add_argument("--state-dir", required=True)
    status.set_defaults(func=cmd_status)

    wait = sub.add_parser("wait", help="Wait until settled or awaiting_decision")
    wait.add_argument("--state-dir", required=True)
    wait.add_argument("--timeout", type=float, default=600)
    wait.set_defaults(func=cmd_wait)

    ns = parser.parse_args(argv)
    return int(ns.func(ns))


if __name__ == "__main__":
    raise SystemExit(main())
