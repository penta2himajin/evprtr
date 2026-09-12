# Findings: Needle contract live smoke (A+B+D+C) — 2026-09-05

Commit: `7808fde` on `main`. Engine: cactus-needle **2.0.10** package + cached `libneedle.dylib` **2.0.3** (`~/.cache/cactus-needle/2.0.3/`). Upstream: oMLX Maple `deepgrove--maple-preview-2bit-mlx` on `:8000`. Compositor `:8741`.

## What was verified

| Check | Result |
|---|---|
| **A** short imperative query | Pass — essay / Maple NL normalized to `Call …` (e.g. `Call ls. path="."` len 17; write brief len 88) |
| **B** ≤5 tools | Pass — 9 OpenAI tools → Needle saw `write, edit, read, ls, grep` |
| **D** `max_new_tokens=256` | Pass — runtime default observed in path smoke |
| **D** `min_confidence=0.35` | Partial — write at **0.41** accepted; empty/`Call ls` abstains still fall through |
| **C** `system` facts | Pass — `date` / `locale: en-US` / `device: coding-agent` bound on path init |

## End-to-end compositor (NL→Needle forced)

`EVPRTR_MAPLE_TOOLS_PRIMARY=0` + `EVPRTR_NEEDLE_VIA_MAPLE_NL=1` so Needle is actually exercised.

| Task | Trace | Outcome |
|---|---|---|
| `ls path="."` (9 tools) | `tr-4eac104b…` | Needle query/tools correct, but **empty call** (`conf=0.0`, reasoning loop / phone-call-ish parse). Fallback Maple-with-tools then emitted `ls path=.` |
| create-file write | `tr-c2aa54f0…` | **Pass** — `needle_via=needle_from_maple_nl`, `write path=live-needle-contract-smoke.txt content=needle-ab-ok`, `confidence=0.4098` |

## Maple-tools-primary (default) observation

Pi readonly (`read,ls,grep`): Maple returned shell-like **prose** without `<tool_call>` / `tool_calls` → path unify presented `maple_final_content` (**Needle not consulted**). Trace `tr-63cae3fc…`. This is expected under the unify policy; Needle only runs on empty/broken markup, not on non-empty prose.

Direct HTTP with markup primary + clear `ls` task: Maple emitted native `tool_calls` itself (`tr-fb27f20d…`) — Needle unused.

## Direct Needle path smoke (no Maple)

- Essay → `Call ls. path="."`, 5 tools, 256 tok, system facts: empty call (phone-call misread of `Call ls`).
- `Call write. path=… content must be exactly: …`: solid structured write (`confidence≈0.82` in one run).

## Verdict

Contract wiring (**A/B/D/C**) is live and observable in traces. **Write structuring is stabilized** under the new defaults. **`Call ls.` remains a weak spot** on this Mac engine build (phone-call / empty-call); insurance still depends on Maple fallback for listing. Default `min_confidence=0.35` did **not** block the live create-file write (0.41).

## Follow-ups (optional)

1. Soften or specialize the `Call ls` / `Call read` surface if empty-call rates stay high (e.g. `ls path="."` without leading `Call` for list tools only).
2. Keep maple-tools-primary as default; treat Needle as write/structure insurance, not primary ls router.
3. Re-smoke Path B write/bash with Needle **on** (markup primary) once listing prose→tools is separately tightened.
