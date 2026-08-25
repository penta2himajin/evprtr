# Findings: Maple+evprtr tinyserve implementation start — 2026-08-25

## Setup

- Maple back on `:8080` (Qwen stopped for VRAM)
- evprtr path B: `EVPRTR_BUFFER_SIDE_EFFECTS=0`
- Approvals via **Pi RPC + `harness/pi/rpc_bridge.py`** (not `-p` / AUTO_APPROVE)
- Plan from Qwen in `tinyserve/docs/implementation-plan.md` (bind `:3000`)

## Commits this session

| Commit | Change |
|---|---|
| `2783f3f` | `EVPRTR_PI_AUTO_APPROVE` for path B `-p` (**removed** in favor of RPC) |
| `91cca3c` | Agentic repair when `tools` present (stop “Emit ONLY answer body”) |
| (pending) | Drop AUTO_APPROVE; add `rpc_bridge.py`; repair primary-task + agentic pseudo-markup; Needle `fallback_maple` reason + query pick |

## What worked

- Path B gate audited and ran real `bash` under the old AUTO_APPROVE experiment
- Stream shim + Maple upstream healthy after swap
- Gate now blocks non-UI (`-p`) and expects TUI or RPC `decide`
- **RPC smoke (2026-08-25):** `write` → `PENDING_APPROVAL` → `decide --approve` → file written (`rpc-ok`)
- Needle `empty_call` under `tool_choice=auto` now **falls back to Maple** (`empty_call_fallback`) instead of ending the Pi turn with “No tool exists…”

## What blocked implementation

1. **Repair / pseudo-tool loop** still turns coding turns into short prose (`Ready` / `Understood`) when structured tools fail — mitigated by agentic repair + primary-task keep.
2. **Needle** often abstains (`empty_call`, conf≈0.03) on Pi `write` schemas — fixed by Maple fallback; structured Needle wins still preferred when it emits calls.
3. **Maple task adherence**: preferred exploratory `bash` / `mkdir` over `write` for Cargo.toml.
4. **Windows + bash redirects**: heredoc write corrupted `Cargo.toml` (restored). Deferred until write/edit path is the active work item.
5. Occasional Pi requests arrived with **`has_tools: false`** when `--tools` was narrowed.

## Next improvements (priority)

1. Drive tinyserve micro-steps through RPC approve loop (gate verified).
2. Needle: act on logged skip reasons when convert/query/confidence still fail.
3. Gate policy (optional, later): on Windows, warn/block `bash` file redirects in favor of `write`/`edit`.
