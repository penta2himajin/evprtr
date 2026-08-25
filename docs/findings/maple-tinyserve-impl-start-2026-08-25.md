# Findings: Maple+evprtr tinyserve implementation start — 2026-08-25

## Setup

- Maple back on `:8080` (Qwen stopped for VRAM)
- evprtr path B: `EVPRTR_BUFFER_SIDE_EFFECTS=0`
- Pi gate + `EVPRTR_PI_AUTO_APPROVE=1` for unattended `-p`
- Plan from Qwen in `tinyserve/docs/implementation-plan.md` (bind `:3000`)

## Commits this session

| Commit | Change |
|---|---|
| `2783f3f` | `EVPRTR_PI_AUTO_APPROVE` for path B `-p` |
| `91cca3c` | Agentic repair when `tools` present (stop “Emit ONLY answer body”) |

## What worked

- Path B auto-approve audited and ran real `bash` (e.g. `ls -la`)
- Stream shim + Maple upstream healthy after swap

## What blocked implementation

1. **Repair / pseudo-tool loop** still turns coding turns into short prose (`Ready` / `Understood`) when structured tools fail.
2. **Needle** still `fallback_maple` often.
3. **Maple task adherence**: preferred exploratory `bash` / `mkdir` over `write` for Cargo.toml.
4. **Windows + bash redirects**: heredoc write corrupted `Cargo.toml` (restored).
5. Occasional Pi requests arrived with **`has_tools: false`** when `--tools` was narrowed.

## Next improvements (priority)

1. Repair: preserve primary user task across multi-turn; don’t replace with “(no user text)”.
2. Pseudo-tool repair: when tools are present, convert or re-ask for structured tool_calls instead of `tool_choice=none` + prose-only.
3. Needle: diagnose empty convert / fallback on Pi schemas.
4. Gate policy (optional): on Windows, warn/block `bash` file redirects in favor of `write`/`edit`.
