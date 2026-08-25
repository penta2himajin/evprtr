# Probe: Pi → evprtr → tinyserve (plan-only) — 2026-08-25

## Setup

- Maple (`llama-server`) on `:8080`, ctx 16K
- evprtr on `:8741` with Needle lib + `EVPRTR_BUFFER_SIDE_EFFECTS=1`
- Pi provider `evprtr` → `http://127.0.0.1:8741/v1` (model id `evprtr`)
- Target cwd: `tinyserve` (scaffold only)

## Blocker fixed mid-run

Pi always sends `stream=true`. Compositor v0 rejected that with 400.

**Fix (committed `a93a322`):** non-streaming completion wrapped as OpenAI SSE (`compositor/api/stream_shim.py`), plus `developer` → `system` role normalize.

## Tool / buffer observations

| Observation | Detail |
|---|---|
| Needle tool path | Trace `tr-bfb2cebaf3bd4c08a2a9734ddb27470b`: `tool_select` start → **`fallback_maple`** (Needle did not own the turn) |
| Proposed tool | Maple (with tools in request) emitted **`bash`** |
| Arguments quality | Degenerate: `ls -la /home/local/local/...` with massive path + prose repetition (verify truncated some repetition) |
| Buffer | **Caught** → `appr-b23b4f3b13324f5ab50e`; harness saw summary text, not an executable tool_call |
| Apply | Rejected manually (`decision_note`: degenerate repeated path) |
| `web_search` | Pi built-ins are `read` / `bash` / `edit` / `write` — **no `web_search` in this run**. Classifier still treats `web_search` as side-effect (unit check: safe=`read`, risky=`web_search`,`bash`) |
| Agent loop | After buffer stripped the only tool_call, Pi stopped (`stopReason=stop`) without a real plan |

## Plan quality

- Pi+tools: no usable plan (buffer interrupt + garbage bash).
- Direct `/v1/chat/completions` without tools: short “plan” with wrong/hallucinated deps (`hyper 0.7`, `axul`, etc.) — **not actionable**.

A supervisor-side recommended plan for the next implement turn is in the session notes / user reply (not agent-authored).

## Follow-ups (compositor)

1. Log **why** Needle returns `None` (empty converted tools vs empty query vs low confidence) on `fallback_maple`.
2. Consider verify on **tool call arguments** (repetition / absurd paths) before enqueue, or auto-reject.
3. When all tool_calls are buffered, present a clearer “pending approval — continue after review” signal so harnesses can keep the loop alive.
4. True token streaming still future; SSE shim is harness-compat only.
