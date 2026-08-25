# Probe: path B Pi gate + tinyserve plan — 2026-08-25

## Shipped

- `harness/pi/evprtr-side-effect-gate.ts` — Pi `tool_call` confirm/block for `bash`/`powershell`/`write`/`edit`; passthrough `read`/`grep`/`find`/`ls`
- `POST /v1/approvals` for harness audit enqueue
- Compositor run with `EVPRTR_BUFFER_SIDE_EFFECTS=0` (`healthz.buffer_side_effects: false`)
- Repo: https://github.com/penta2himajin/evprtr (`e73882c` + follow-up findings)

## Plan-only run (files attached)

- Pi loaded the gate extension; scaffold files were inlined via `@…`
- Model returned a prose plan in one turn (**no structured `tool_calls`**), so the gate did not fire
- Plan quality: weak / confused (`HTTPx` as server stack, `http` crate with invented features). Treat as validation of the loop, not as an actionable build plan
- `stopReason: length` (truncated)

## Forced-bash check (`--tools bash`)

- Maple/Needle path again failed to emit OpenAI `tool_calls`; content contained pseudo `</tool_call>` markup instead
- Therefore Pi never entered `tool_call` → gate not invoked; no new `pi_gate` approval rows
- Reinforces: path B gates **harness execution**; compositor verify still needed for pseudo-markup / non-structured tool attempts

## Smoke

- `POST /v1/approvals` with a bash payload succeeded (manual smoke id `appr-9d102c1acf9b43bd970e`)

## Next for gate confidence

1. Interactive TUI run where Maple/Needle actually returns `tool_calls` for `bash`, confirm dialog + audit approve/reject
2. Or a unit/integration test that drives Pi RPC `extension_ui_request` / mock tool_call (heavier)
3. Keep improving Needle routing so `--tools bash` yields real `tool_calls`
