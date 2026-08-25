# Pi harness integration (path B)

Recommended when operating evprtr as a **normal coding agent** behind Pi:

1. Point Pi at evprtr (`~/.pi/agent/models.json` provider `evprtr` → `http://127.0.0.1:8741/v1`).
2. **Disable** the compositor response-side buffer so `tool_calls` reach Pi:
   ```text
   EVPRTR_BUFFER_SIDE_EFFECTS=0
   ```
3. Load the side-effect gate extension:
   ```powershell
   pi -e C:\path\to\evprtr\harness\pi\evprtr-side-effect-gate.ts --provider evprtr --model evprtr
   ```

## What the gate does

| Tools | Behavior |
|---|---|
| `read`, `grep`, `find`, `ls` | Passthrough (no prompt) |
| `bash`, `powershell`, `write`, `edit` | Confirm in TUI/RPC; blocked in `-p` / JSON (no UI) |

Approved calls execute via **Pi’s built-in tools** (same agent loop, results return to the model). Optional audit posts to `POST /v1/approvals` when the compositor is reachable (`EVPRTR_BASE_URL`, default `http://127.0.0.1:8741`).

## vs compositor buffer (path A)

| | Path B (this) | Path A (`EVPRTR_BUFFER_SIDE_EFFECTS=1`) |
|---|---|---|
| Gate location | Pi `tool_call` hook | Strip `tool_calls` in API response |
| Execution | Pi harness | External apply (v0 record-only) |
| Session continuity | Natural | Broken unless results re-injected |

Do not enable A and B together for the same run: A prevents side-effect `tool_calls` from reaching Pi, so B never sees them.
