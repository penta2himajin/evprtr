# evprtr

[日本語](./README.ja.md)

Local **composite layer** that calls model runtimes and composes them into one OpenAI-compatible model face — so coding harnesses (OpenCode, Codex, Pi, …) can do practical agent work without relying on frontier cloud models.

The name **evprtr** comes from **evaporator**, the evaporator used when turning maple sap into syrup. Capability is concentrated by mechanism, not by distilling weights. Early focus is Maple-Preview plus helpers such as Needle. Details: [docs/architecture.md](./docs/architecture.md).

## Status

OpenAI-compatible API (`stream=true` is handled by a non-streaming SSE shim for harnesses like Pi), failure traces, verify/repair (repetition + pseudo tool markup), a peelable Needle 2 tool-determination path, and a side-effect tool approvals buffer. Maple remains the prose/reasoning runtime.

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
| `tests/` | TDD suite |
| `docs/architecture.md` | Layers, naming, boundaries |
| `docs/overview.md` | Short user-facing intro |

## License

MIT. See `LICENSE`.
