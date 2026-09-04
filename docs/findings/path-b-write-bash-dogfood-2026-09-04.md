# Findings: Path B write/bash approval dogfood — 2026-09-04

Branch: `cursor/maple-gbnf-no-needle`. Needle off, markup primary, `BUFFER_SIDE_EFFECTS=0`.

## Setup

- Compositor `:8741` → oMLX Maple
- `harness/pi/rpc_bridge.py run` + `evprtr-side-effect-gate.ts`
- Isolated cwd: `.evprtr/dogfood-side-effect/`
- Tools: `read,ls,grep,write,bash`

## What worked

| Step | Result |
|---|---|
| Gate confirm for `write` | Approved via RPC; file created |
| Gate confirm for `bash wc -l` | Approved via RPC |
| Audit `POST /v1/approvals` | Records with `approved` / `rejected` + duplicate notes |
| Duplicate suppression | Re-`write` same content / same `bash` → auto-reject (`duplicate` tag) |

File on disk: `hello_evprtr.txt` with the requested two lines.

## What hurt the long turn

1. **First prompts failed before the gate**
   - `ls`-first → `missing_mutation` stripped tools; Pi saw `toolUse` with empty tools.
   - File-body prose dump → `maple_prose_stop` (mutation task treated as answer).
2. **After successful write+bash**, the agent kept proposing more writes/bashes
   (`cat`, `echo`, duplicate write). Gate correctly blocked duplicates; run did not
   settle cleanly into a final prose stop within the supervisor window.

## Compositor fixes applied during dogfood

- Skip `maple_prose_stop` when `task_wants_mutation`.
- On Needle-off empty/thin mutation miss, try `synthetic_mutation_response`.
- Sanitize `missing_mutation` / degenerate → `finish_reason=stop` (no empty `toolUse`).

## Verdict

Path B **approval + audit works** for write/bash. Markup path **can** emit gated
mutations. Long-turn **stop discipline** after success is still weak under Needle-off
Maple; Needle (or stronger settle heuristics) still looks useful for agent loops.
