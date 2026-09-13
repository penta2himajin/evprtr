# Findings: momij serve-default fix (Pi → evprtr → momij) — 2026-09-13

## Context

The 2026-09-12 Pi smoke (P0-E, `--backend mlx`) broke after momij `ffe44eb`
(default FlashHead SuffixSpec on). Root causes were momij-side; see
`momij/docs/findings-evprtr-serve-default-fix-2026-09-13.md` (momij `84df620`).

1. mlx backend: SuffixSpec verify re-prefilled the full prefix per draft step →
   ~4.8k-token prompts never returned (>240 s).
2. seedless default backend: FlashHead approximation drifted off exact greedy on
   the tool-markup prompt → degenerate repetition, no tool_calls.
3. Engine-level: SuffixSpec batch branch could overshoot `max_tokens`.

## Fix (momij `84df620`)

- Exact `lm_head` is the default verify head; FlashHead opt-in
  (`MOMIJ_EXACT_HEAD=0`).
- SuffixSpec chain appends clamp to the `max_tokens` budget (lossless verified
  on real text, incl. forced batching + M-row chain evals).
- MLX backend ignores `useSuffixSpec` (no hang).

## Verified with evprtr (compositor default stack, Needle off, markup primary)

| Check | Result |
|---|---|
| short tools (markup → tool_calls `ls`) | PASS 0.7 s |
| Pi-like 3,985-tok prompt → `ls` + `grep` tool_calls | PASS 19.1 s wall |
| plain prose | PASS |
| mlx backend 4,822-tok prompt | PASS 7.6 s (no hang) |
| seedless bench (p128/g256) | decode 190.7 tok/s; chain K=8 365.6 tok/s match=true |

No evprtr changes required; the compositor behaved identically before/after.