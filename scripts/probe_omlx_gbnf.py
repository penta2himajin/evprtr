#!/usr/bin/env python3
"""Probe oMLX grammar / structured_outputs against a live server.

Usage:
  export OMLX_BASE_URL=http://127.0.0.1:8000/v1
  export OMLX_API_KEY=...   # or rely on ~/.omlx/settings.json auth.api_key
  export OMLX_MODEL=deepgrove--maple-preview-2bit-mlx
  python scripts/probe_omlx_gbnf.py

Exit 0 if constrained fields clearly differ from unconstrained control.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path


def _api_key() -> str:
    env = os.environ.get("OMLX_API_KEY", "").strip()
    if env:
        return env
    settings = Path.home() / ".omlx" / "settings.json"
    if settings.is_file():
        data = json.loads(settings.read_text(encoding="utf-8"))
        key = (data.get("auth") or {}).get("api_key")
        if key:
            return str(key)
    raise SystemExit("set OMLX_API_KEY or configure ~/.omlx/settings.json auth.api_key")


def chat(base: str, key: str, body: dict) -> tuple[int, dict]:
    req = urllib.request.Request(
        f"{base.rstrip('/')}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as res:
            return res.status, json.loads(res.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            return exc.code, json.loads(raw)
        except json.JSONDecodeError:
            return exc.code, {"error": raw}


def content_of(resp: dict) -> str:
    choices = resp.get("choices") or []
    if not choices:
        return ""
    msg = choices[0].get("message") or {}
    return str(msg.get("content") or "")


def main() -> int:
    base = os.environ.get("OMLX_BASE_URL", "http://127.0.0.1:8000/v1")
    model = os.environ.get("OMLX_MODEL", "deepgrove--maple-preview-2bit-mlx")
    key = _api_key()
    prompt = "Say hello in a full English sentence about the weather today."

    cases: list[tuple[str, dict]] = [
        ("control", {}),
        ("structured_outputs.grammar", {"structured_outputs": {"grammar": 'root ::= "ZQX"'}}),
        ("guided_grammar", {"guided_grammar": 'root ::= "ZQX"'}),
        ("top_level_grammar", {"grammar": 'root ::= "ZQX"'}),
        ("structured_outputs.regex", {"structured_outputs": {"regex": "ZQX"}}),
        ("structured_outputs.choice", {"structured_outputs": {"choice": ["ZQX", "WYP"]}}),
        (
            "structured_outputs.grammar_rules",
            {
                "structured_outputs": {
                    "grammar": 'root ::= color\ncolor ::= "amber" | "ivory"'
                }
            },
        ),
    ]

    results: dict[str, str] = {}
    print(f"base={base} model={model}")
    for label, extra in cases:
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 32,
            "temperature": 0.9,
        }
        body.update(extra)
        status, resp = chat(base, key, body)
        text = content_of(resp) if status < 400 else f"HTTP {status}: {resp}"
        results[label] = text
        print(f"{label}: {text!r}")

    control = results.get("control", "")
    constrained_ok = all(
        results.get(k, "").strip() in {"ZQX", "WYP", "amber", "ivory"}
        or results.get(k, "").strip().startswith("ZQX")
        for k in (
            "structured_outputs.grammar",
            "guided_grammar",
            "structured_outputs.regex",
            "structured_outputs.choice",
            "structured_outputs.grammar_rules",
        )
    )
    top_ignored = "ZQX" not in results.get("top_level_grammar", "") and len(
        results.get("top_level_grammar", "")
    ) > 10
    control_free = "ZQX" not in control and len(control) > 10

    print("---")
    print(f"control_unconstrained={control_free}")
    print(f"constrained_fields_ok={constrained_ok}")
    print(f"top_level_grammar_ignored={top_ignored}")

    if control_free and constrained_ok and top_ignored:
        print("PASS: oMLX enforces structured_outputs.grammar / guided_grammar / regex / choice")
        return 0
    print("FAIL: unexpected oMLX grammar behavior")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
