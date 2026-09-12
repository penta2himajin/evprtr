# Findings: momij as Maple upstream (P0-E) — 2026-09-12

## Setup (final smoke)

- momij: release `serve --backend mlx --port 8744 --model-id maple-preview`
  - tip: `a3fe719` on `cursor/omlx-dropin-api`
- compositor: `:8741` → `http://127.0.0.1:8744/v1`, Needle off, markup primary, grammar off
- Pi: `--provider evprtr --model evprtr --tools read,ls,grep`

## Results

| Check | Result |
|---|---|
| API drop-in (P0 A–D) | OK |
| Short markup → `<tool_call>` | OK |
| Long prompt collapse (`prompt_tokens ≳ 600`) | **Fixed** by SWA RoPE + sliding mask (`a3fe719`); see `docs/findings-long-prompt-swa-2026-09-12.md` in momij |
| Pi readonly smoke (ls → grep → prose) | **PASS** (2026-09-12 evening) |

### Pi smoke (final)

- `toolCall ls` → ok
- `toolCall grep Overview AGENTS.md` → ok
- final text `## Overview`
- Traces: `maple_tool_markup_attached`, `pseudo_tool_promoted` / `maple_tools_primary`; prompt_tokens ~4.7–4.9k

## Earlier failures (superseded)

- seedless: Metal `endEncoding` assert + garbage quality (mitigated in `8b47021`; long-prompt parity still open)
- mlx long Pi system: SWA offset clamp (fixed in `a3fe719`)

## Still open

- Default serve backend remains seedless — prefer **mlx** for agentic until seedless long-prompt parity is re-checked
- xgrammar perf on long prompts (P1 insurance path)
- Optional: attach `TOOLS_GRAMMAR` before markup strips `tools` (evprtr order)
