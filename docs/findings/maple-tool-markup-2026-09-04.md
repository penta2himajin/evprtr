# Findings: Maple ``<tool_call>`` markup primary — 2026-09-04

Branch: `cursor/maple-gbnf-no-needle`.

## Decision (DeepGrove chat → evprtr)

DeepGrove’s stable tool path is **not** “model-native OpenAI `tool_calls` alone”.
It is gateway shaping: fixed prompt contract (`<tools>` / `<tool_call>` from the
HF `chat_template`) + parse + separate content vs tool channel for the client.

evprtr adopts that **inside the compositor**; harness execution stays OpenAI
`tool_calls`.

| Layer | Role |
|---|---|
| Primary | `maple_tool_markup.v1` — inject `<tools>`, rewrite history, parse `<tool_call>` → OpenAI `tool_calls`; plain prose stays `content` |
| Insurance | `EVPRTR_TOOLS_GRAMMAR=1` — whole-completion `structured_outputs` (default **off**) |
| Repair | Needle structure / degenerate correction (keep; do not delete) |

## Modules

- `compositor/tools/maple_tool_markup.py` — prompt + history rewrite
- `compositor/tools/pseudo_tool.py` — parser (HF `arguments` key + flat objects)
- `core._maple_tools_primary_path` — attach markup before Maple call

## Flags

| Flag | Default | Meaning |
|---|---|---|
| `EVPRTR_MAPLE_TOOL_MARKUP` | `1` | Markup primary |
| `EVPRTR_TOOLS_GRAMMAR` | `0` | JSON grammar insurance |
| `EVPRTR_NEEDLE_ENABLED` | (lib) | Repair / structure fallback |

## Out of scope

- DeepGrove-only public API
- Hardcoding two tools (`search`/`python`) in production
- Dropping tool results from the conversation

## Live smoke (Needle off, grammar off, markup on)

Pi `--tools read,ls,grep` against evprtr `:8741` → oMLX Maple:

| Step | Result |
|---|---|
| `ls` | executed with absolute repo path |
| `grep Overview AGENTS.md` | executed |
| `read AGENTS.md` | executed |
| final | `maple_final_content` plain text gist; `EXIT:0`; no tool loop |

Trace phases consistently include `maple_tool_markup_attached`. No `tools_grammar_*`.
