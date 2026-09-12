# Findings: momij as Maple upstream (P0-E) — 2026-09-12

## Setup

- momij: `serve --model ~/models/deepgrove/maple-preview-2bit-mlx --model-id maple-preview --port 8742 --backend mlx` (release binary)
- compositor: `EVPRTR_UPSTREAM_BASE_URL=http://127.0.0.1:8742/v1`, `EVPRTR_UPSTREAM_MODEL=maple-preview`, Needle off, maple-tools-primary + markup on, `:8741`
- Pi: `--provider evprtr --model evprtr`

## Results

| Check | Result |
|---|---|
| `/healthz`, `/v1/models`, chat shape | OK |
| Extra keys (`tools`, …) ignored (no 400) | OK (P0-B) |
| Short markup → `<tool_call>` (momij direct / compositor) | OK (`pseudo_tool_promoted`) |
| Pi readonly smoke (full Pi system ≈2.4k prompt toks) | **FAIL** — no tool_calls; degeneration |
| `--backend seedless` | **Unusable** for E: Metal assert crash under load; generate quality garbage |

### seedless

First serve crashed mid-Pi-retry:

`-[_MTLCommandEncoder dealloc]: failed assertion 'Command encoder released without endEncoding'` (exit 134)

CLI `generate --backend seedless` also emits nonsense (`way way way-1-1-1`). Use **mlx** for agentic smoke until seedless is fixed.

### mlx + short tools

Compositor request with one `ls` tool, short user text → `finish_reason=tool_calls`, trace `needle_via=pseudo_tool_promoted`, `maple_tool_markup_attached`.

### mlx + Pi system

Captured Pi body: `stream=true` (compositor shim → non-stream), `max_tokens=8192`, system ≈8931 chars (~2.2k toks) including “Available tools:” prose (not `<tools>` XML).

Length sweep (Pi system prefix + markup suffix → momij mlx, `temperature=0`):

| ~prompt_tokens | `<tool_call>` | Notes |
|---|---|---|
| 354 | yes | truncated system 500 chars |
| 508 | no | coherent reasoning, no call |
| ≥613 | no | loops / raw `<\|im_start\|>` text |

Full Pi prompt never emits markup; compositor presents `maple_final_content`.

## Verdict

**API drop-in (P0 A–D contract) is enough for short harness-shaped calls.**  
**P0-E Pi smoke is blocked on momij mlx long-prompt degeneration** (and seedless instability), not on evprtr wiring.

Next: fix long-context / sampling / special-token stop on momij, or re-smoke when seedless is healthy; P1 grammar may help force `<tool_call>` once decode is constrained.
