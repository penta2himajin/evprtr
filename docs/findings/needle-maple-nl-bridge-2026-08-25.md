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
- Hole fixes after smoke4/5: NL rejects degenerate args (`needle_quality_miss` → user-task retry); Needle correct validates before accept; create-file path/body align; empty edit texts flagged; `_finish_tool_result` uses full verify loop.
- Live re-smoke: `tr-61045121…` (smoke6) Pass — quality miss then retry → exact `write path=live-nl-smoke6.txt content=nl-needle-ok`.
- Further: deterministic `synthetic_create_file` / `synthetic_edit_replace` when Needle still fails on explicit prompts; create-file body stops at harness blurbs.
- Live `tr-e2921bda…` / `tr-f5f655d2…` (smoke7): create write exact; edit via=`synthetic_edit_replace` with correct old/new.
