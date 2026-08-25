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

## Read-only settle — implemented

Observed: Maple `read`×N then `stop` with no `write`.

**Shipped:** peelable `MissingMutationDetector` (`maple_preview.missing_mutation`) when:

- request offers mutation tools (`write`/`edit`/`bash`/`powershell`)
- user task matches mutation intent (`task_wants_mutation`)
- response has no mutating `tool_calls`

Sanitize drops the read-only calls; repair / Needle correct uses `user_task_instruction` (imperative create-file brief). No AUTO_APPROVE.

Deferred: Pi harness steer after N reads; `tool_choice=required`.

## Tests

`tests/test_needle_nl_bridge.py` covers fence extract, chunk plans, NL→chunk path, degenerate→Needle correct.

## Live evaluation (2026-08-25)

| Probe | Result |
|---|---|
| Before harden (`tr-231f5aa2…`) | Fail: Maple NL 11k runaway; Needle empty_call (“flight delayed…”); Maple fallback no tools |
| Needle direct imperative | Pass: conf≈0.96 correct `write` |
| Soft “convert this task” | Weak: abstain / wrong tool |
| After NL clamp (`tr-0e6fb7f4…`) | Partial: path OK but Needle returned `read` + absurd path |
| Mutation-miss + create-file normalizer (`tr-ceb08346…`) | **Pass**: empty_call → `needle_retry_user_task` → `write path=live-nl-smoke3.txt content=nl-needle-ok` |

**Verdict:** Small-write bridge is viable with clamp + user-task retry/normalizer. Soft Maple NL alone is not enough. Chunk/long-content still needs more live coverage before calling it done.

## Follow-up (same day)

- Live small-write OK → proceeded with read-only settle detector (above).
- Unit: `tests/test_degenerate_tool_args.py` covers missing_mutation diagnose + sanitize.
