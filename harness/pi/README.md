# Pi harness integration (path B)

Recommended when operating evprtr as a **normal coding agent** behind Pi:

1. Point Pi at evprtr (`~/.pi/agent/models.json` provider `evprtr` → `http://127.0.0.1:8741/v1`).
2. **Disable** the compositor response-side buffer so `tool_calls` reach Pi:
   ```text
   EVPRTR_BUFFER_SIDE_EFFECTS=0
   ```
3. Load the side-effect gate and drive approvals over **Pi RPC** (not `-p` auto-approve):

```powershell
# A: run agent (blocks on each gated tool until decide)
python harness/pi/rpc_bridge.py run `
  --cwd C:\Users\penta\repos\tinyserve `
  --state-dir C:\Users\penta\repos\tinyserve\.evprtr\pi-rpc `
  --fresh-state `
  --provider evprtr --model evprtr `
  --prompt-file C:\Users\penta\repos\tinyserve\.evprtr\prompts\step.txt

# Prefer --prompt-file on Windows PowerShell (quoting breaks multiline --prompt).

# B: supervisor / Cursor
python harness/pi/show_pending.py --state-dir ...\pi-rpc
python harness/pi/rpc_bridge.py pending --state-dir ...\pi-rpc
python harness/pi/rpc_bridge.py decide --state-dir ...\pi-rpc --id <uuid> --approve
python harness/pi/rpc_bridge.py wait --state-dir ...\pi-rpc
```

Cursor / Auto supervisors: poll `show_pending.py`, read `message` / tool args, then `decide --approve` or `--reject`. Do not use `pi -p` for mutation dogfood — side-effect tools are blocked without UI/RPC.
### Rewrite / reject loop suppression

| Layer | Behavior |
|---|---|
| Pi gate | After a write/edit/shell is **approved once**, an identical mutation is **blocked** immediately (no second confirm). Confirm body includes `fp=...`. |
| `rpc_bridge` | Same `fp` already approved → auto-reject + **abort**. Identical pending twice → auto-reject; three times → abort. Tunable: `--identical-pending-reject-at` / `--identical-pending-abort-at`. |

Interactive TUI (`pi` without `--mode rpc`) also works: the gate uses `ctx.ui.confirm` in-process.

## What the gate does

| Tools | Behavior |
|---|---|
| `read`, `grep`, `find`, `ls` | Passthrough (no prompt) |
| `bash`, `powershell`, `write`, `edit` | Confirm in TUI or via RPC bridge; **blocked** in `-p` / JSON (no UI) |

Approved calls execute via **Pi’s built-in tools** (same agent loop, results return to the model). Audit posts to `POST /v1/approvals` when the compositor is reachable (`EVPRTR_BASE_URL`, default `http://127.0.0.1:8741`).

## vs compositor buffer (path A)

| | Path B (this) | Path A (`EVPRTR_BUFFER_SIDE_EFFECTS=1`) |
|---|---|---|
| Gate location | Pi `tool_call` hook | Strip `tool_calls` in API response |
| Execution | Pi harness after confirm | External apply (v0 record-only) |
| Session continuity | Natural | Broken unless results re-injected |

Do not enable A and B together for the same run: A prevents side-effect `tool_calls` from reaching Pi, so B never sees them.
