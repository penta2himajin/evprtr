# evprtr

[日本語](./README.ja.md)

Local **composite layer** that calls model runtimes and composes them into one OpenAI-compatible model face — so coding harnesses (OpenCode, Codex, Pi, …) can do practical agent work without relying on frontier cloud models.

The name **evprtr** comes from **evaporator**, the evaporator used when turning maple sap into syrup. Capability is concentrated by mechanism, not by distilling weights. Early focus is Maple-Preview plus helpers such as Needle. Details: [docs/architecture.md](./docs/architecture.md).

## Status

OpenAI-compatible API (`stream=true` SSE shim), failure traces, verify/repair, Needle tool path, compositor side-effect buffer (path A), and a **Pi harness gate** (path B: `harness/pi/`). Maple remains the prose/reasoning runtime.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -e ".[dev]"

# Upstream = a runtime that already speaks OpenAI chat completions (e.g. llama-server)
set EVPRTR_UPSTREAM_BASE_URL=http://127.0.0.1:8080/v1
set EVPRTR_NEEDLE_LIB_PATH=F:\LLM\models\needle2\libneedle.dll
python -m compositor

# Point a harness at http://127.0.0.1:8741/v1  (model id: evprtr)
```

```bash
pytest
```

## Layout

| Path | Role |
|---|---|
| `compositor/` | Composite layer package (API + orchestration) |
| `harness/pi/` | **Path B** — Pi side-effect gate + RPC approval bridge |
| `tests/` | TDD suite |
| `docs/architecture.md` | Layers, naming, boundaries |
| `docs/overview.md` | Short user-facing intro |

## Path B (Pi RPC approvals) — other machines

Use this when running a normal coding agent behind Pi without AUTO_APPROVE:

1. Install deps / start compositor (see Quick start). Set `EVPRTR_BUFFER_SIDE_EFFECTS=0`.
2. Point Pi at `http://127.0.0.1:8741/v1` (model id `evprtr`).
3. Drive gated `write` / `edit` / `bash` confirms with:

```powershell
python harness/pi/rpc_bridge.py run --cwd <project> --state-dir <project>/.evprtr/pi-rpc --fresh-state --provider evprtr --model evprtr --prompt-file <prompt.txt>
python harness/pi/rpc_bridge.py pending --state-dir <project>/.evprtr/pi-rpc
python harness/pi/rpc_bridge.py decide --state-dir <project>/.evprtr/pi-rpc --id <uuid> --approve
```

Details, rewrite-loop suppression, and tool policy: [`harness/pi/README.md`](./harness/pi/README.md).

## License

MIT. See `LICENSE`.
