# Architecture

## What evprtr is

**evprtr** is a **composite layer** (also: *compositor*). It sits between coding-agent harnesses and local model runtimes. It does **not** run weights itself; it **calls** underlying runtimes and composes their outputs so that weak local models, used together with inference-time mechanisms, approach the practical utility of strong commercial models.

Distillation and weight-training are out of scope by default. Gains come from mechanism: role split, structured tool calling, verification loops, recovery, and observability for debugging.

## Name

**evprtr** is short for **evaporator** — the evaporator used when concentrating maple sap into syrup. The project thickens local model capability the way an evaporator thickens sap; the primary local brain in early work is Maple-Preview.

## Layer vocabulary

| Layer | Name | Role |
|---|---|---|
| Upper | **harness** (OpenCode, Codex, Pi, …) | Agent loop: edits, terminal, session, tools UI |
| Middle | **compositor / composite layer** (this repo) | Calls runtimes, composes a single strong model face |
| Lower | **runtime** | Literal inference engines (e.g. Maple llama.cpp fork, Needle engine) |

Do not call the middle layer a “runtime” or a “harness”. Those words belong to the layers above and below.

## External interface

Harnesses talk to evprtr over an **OpenAI-compatible HTTP API** (`/v1/models`, `/v1/chat/completions`, …). From the harness’s point of view, evprtr is one model id.

## Internal shape (target)

```
Harness  --OpenAI compatible-->  Compositor
                                    ├─ Maple runtime   (reasoning / planning)
                                    ├─ Needle runtime  (tool calls / structured JSON)
                                    ├─ verify / repair loops
                                    └─ traces for debug & improvement feedback
```

Evaluation exists to capture **where** a failure happened (plan / tool select / execute / verify), not to chase public leaderboard scores.

## Failure traces

Every `/v1/chat/completions` attempt gets a trace id (`X-Evprtr-Trace-Id`).

| Piece | Purpose |
|---|---|
| `locus` | Which stage failed: `accept`, `upstream`, `present` (later: plan / tool_select / execute / verify / repair), or `none` |
| events | Ordered stage markers (`ok` / `error`) with timestamps |
| request/response summaries | Counts, roles, finish reasons — **not** full message bodies |
| persistence | JSON under `EVPRTR_TRACE_DIR` (default `.evprtr/traces/`) |
| HTTP | `GET /v1/traces`, `GET /v1/traces/{id}` |

Traces are for debug and improvement feedback when practical runs go wrong.

## Verify / repair

The compositor runs an injectable **verify stack** (`VerifyBundle`):

| Layer | Module | Role |
|---|---|---|
| Loop | `compositor/core.py` + `verify/pipeline.py` | Diagnose → sanitize → repair; policy-agnostic |
| Generic | `verify/detectors_generic.py`, `verify/repair.py` (`FreshConstrainedRepair`) | Repetition detect/truncate; fresh constrained re-call |
| Maple-Preview (peelable) | `verify/detectors_maple_preview.py`, `MaplePreviewRepair` | Thin preamble; empty+long reasoning; **pseudo `<tool_call>` markup in content**; bullet-oriented repair wording |

Default bundle is `VerifyBundle.maple_preview()`. Use `VerifyBundle.generic_only()` to drop Maple heuristics. Each diagnosis carries `policy_id` in traces so peel/generalize decisions are observable.

Near-term target remains Maple-Preview; mid-term other small models should add/replace peelable policies rather than forking the loop.

## Needle 2 tool path

When the request includes `tools` and `tool_choice` is not `none`, the compositor may route **tool determination** to Needle 2 (`needle2.tool_path`) instead of Maple:

| Outcome | Behaviour |
|---|---|
| `function_calls` non-empty | OpenAI-shaped `tool_calls` response (`finish_reason=tool_calls`) |
| Empty call `[]` | Structured prose refusal (no `<tool_call>` markup) — covers schema conflicts / no matching tool |
| `tool_choice=none` | Needle skipped; Maple + pseudo-markup verify still apply |
| Needle unavailable / low confidence | Fall back to Maple |

Engine default path on the author machine: `F:/LLM/models/needle2/libneedle.dll` (`EVPRTR_NEEDLE_LIB_PATH` / `NEEDLE_LIB_PATH`). Disable with `EVPRTR_NEEDLE_ENABLED=0`.

Trace stage: `tool_select`. Summary flags: `needle_tool_path`, `needle_empty_call`, `needle_confidence`.

## Side-effect buffer / approvals

Risky tool calls (`write`, `bash`, `web_search`, …) are **not** returned to the harness for immediate execution. They are enqueued:

| API | Role |
|---|---|
| `GET /v1/approvals` | List pending/decided actions |
| `POST /v1/approvals/{id}/approve` / `reject` | Human/supervisor decision |
| `POST /v1/approvals/{id}/apply` | v0: record apply only (no host exec yet) |

Safe tools (`read`, `grep`, …) still pass through. Disable buffering with `EVPRTR_BUFFER_SIDE_EFFECTS=0`. Header `X-Evprtr-Approvals` lists enqueued ids.

Trace stages: `verify`, `repair`, `approval_buffer`. Response summary includes `repetition_repaired`, `buffered_approvals`.

## Local model paths (author machine)

Default Maple artifacts and servers live under `F:/LLM` (see that tree’s README). evprtr configures upstream base URLs; it does not vendor GGUF weights.
