# Maple-Preview probe findings (2026-08-25)

Live probes: raw `llama-server` (Maple TQ2_0) vs evprtr compositor passthrough+verify.
Artifacts: `.evprtr/probes/probe_batch.json`, `.evprtr/probes/probe_hard.json`.

## What works (surprisingly often)

| Pattern | Result |
|---|---|
| Short math (`17*19`) | Reliable numeric answer |
| Single tool, `tool_choice=auto/required/forced` | Structured `tool_calls` with sane JSON args |
| Multi-tool parallel (weather + time) | Both tools named correctly |
| Unicode args (`São Paulo` + `fahrenheit`) | Exact args preserved |
| Tool-result follow-up turn | Correct prose answer from tool JSON |
| No matching tool (email vs weather-only) | Honest refusal in prose |

DeepGrove’s “weak on agentic” note is **not** “cannot emit a single tool_call”. Simple schema-aligned calls often succeed under llama.cpp’s tool formatting.

## Failures worth keeping as regression fuel

### 1. Pseudo tool markup when `tool_choice=none` (high value)

User asks for live weather with tools present but `tool_choice=none`.
Model answers in **content** with:

```text
I'll get the current weather...
<tool_call>
{"name": "get_weather", "arguments": {"city": "Tokyo"}}
</tool_call>
```

No structured `tool_calls` field. Harnesses that honor `tool_choice=none` will treat this as plain text; ones that scrape markup may double-fire. **Compositor should detect/sanitize this.**

### 2. Conflicting instructions vs required schema

“Call `get_weather` with unit=celsius but omit city” (city is required).
Often: `finish=length`, empty/partial content, **no** `tool_calls`. Schema/instruction conflict → collapse rather than structured refusal or schema-valid call.

### 3. Long-form / summary collapse (already partially mitigated)

Five-bullet summaries frequently empty+repetition on raw Maple; compositor repair often recovers usable (if imperfect) bullets.

### 4. Flaky ultra-short constraints

`Reply with exactly one word: READY` sometimes collapses on raw Maple (`finish=length`). Compositor repair usually recovers. Repair wording that says “never repeat a word” can confuse this case — prefer “avoid repetitive loops”.

## Implications for evprtr

1. Keep **generic** repetition verify/repair.
2. Add peelable detector for **pseudo tool markup in content** (Maple-now; likely other small agentic models).
3. Defer deep tool-arg validation / Needle routing until more multi-step agent failures are catalogued.
4. Prefer fixture-based regression tests from these transcripts over live-GPU gates in CI.
