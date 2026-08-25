# Findings: Maple NL → Needle structure + chunk writes — 2026-08-25

## Intent

- Maple emits **natural-language** tool instructions (no tool channel).
- Needle turns that NL into OpenAI `tool_calls`.
- Long file bodies: **mechanism splits**, write+edit sentinel plan applies (Needle probed; deterministic fallback).
- Degenerate Maple/Needle args: **Needle correction** before Maple prose repair.

## Flags (default on when Needle is enabled)

| Env | Role |
|---|---|
| `EVPRTR_NEEDLE_VIA_MAPLE_NL` | Maple NL → Needle (default `1`) |
| `EVPRTR_NEEDLE_CHUNK_WRITES` | Fenced-file chunk apply (default `1`) |
| `EVPRTR_NEEDLE_CORRECT_DEGENERATE` | Post-hoc Needle fix (default `1`) |
| `EVPRTR_NEEDLE_MAX_NEW_TOKENS` | Needle budget (default `1024`; engine default was 256) |

## Read-only settle (open consultation)

Observed: Maple `read`×N then `stop` with no `write`. Options (not mutually exclusive):

1. **Compositor diagnose** `tools_offered_but_no_side_effect` when the user task clearly requires write/edit and the model only used read/passthrough — force Maple-NL→Needle or repair. Keeps audit; no AUTO_APPROVE.
2. **Harness policy** (Pi extension): after N reads without a gated tool on an “implement” prompt, inject a user steer (“emit write now”). Session-local, transparent.
3. **`tool_choice=required`** for implement micro-steps — harsh; risks garbage forced calls (need degenerate detect).
4. **Supervisor micro-prompts** that only ask for one write — helps but does not fix model drift alone.

Recommendation to try first: (1) peelable detector + NL→Needle re-ask, then (2) if still weak.

## Tests

`tests/test_needle_nl_bridge.py` covers fence extract, chunk plans, NL→chunk path, degenerate→Needle correct.
