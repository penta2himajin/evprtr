# Findings: maple-tools-primary path unification — 2026-09-04

Branch: `cursor/maple-gbnf-no-needle`.

## Decision

Collapse post-Maple branching around the markup primary path:

| State | Action |
|---|---|
| native / parsed `<tool_call>` | verify → present |
| broken `<tool_call>` | Needle repair (insurance) |
| non-empty prose, no tools | present as final content |
| empty / thin | Needle structure (insurance) |

## Retired

- `maple_prose_stop` special phase (replaced by `maple_final_content`)
- Keyword `task_wants_mutation` gating on this path / NL stop / NL accept
- Default-bundle `MissingMutationDetector`
- `synthetic_mutation_response` (stub returns `None`; helpers kept for later delete)

## Kept

- Markup primary + pseudo promote
- Needle for empty/thin, broken markup, degenerate-arg correction
- Verify shape checks (repetition, degenerate args, thin/empty heuristics)
- Grammar insurance (`EVPRTR_TOOLS_GRAMMAR=0` default)

## Live re-check (Needle off, same day)

Readonly Pi `--tools read,ls,grep`: `EXIT:0`; `ls` + `grep Overview`; short final text.

Path B RPC write/bash (isolated cwd): approved `write hello_unify.txt` then `bash wc -l`; file on disk; **2** approval records (no rewrite spiral in this run); bridge settled.
