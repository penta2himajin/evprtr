# evprtr

## Overview

evprtr is a **composite layer** (compositor): it calls local model **runtimes** and composes them into one OpenAI-compatible model face for coding **harnesses** (OpenCode, Codex, Pi, …). Goal: practical utility comparable to strong commercial models, via mechanism — not distillation.

Name origin: **evaporator** (maple-sap evaporator). See @docs/architecture.md.

## Project Structure

```
compositor/          # composite layer package (this is not a weight runtime)
  api/               # OpenAI-compatible HTTP surface
  runtimes/          # clients that call underlying runtimes
tests/               # TDD suite; run with pytest
docs/                # architecture + overview (engineering docs English-only except overview.ja.md)
```

Do not vendor large GGUF/weights in this repo. Configure upstream URLs to machines/paths such as `F:/LLM`.

## Development Setup

Python 3.11+.

```bash
python -m venv .venv
pip install -e ".[dev]"
git config core.hooksPath git-hooks
```

Upstream runtime must already expose OpenAI-compatible chat completions (e.g. Maple `llama-server`).

```bash
# Windows cmd/PowerShell example
set EVPRTR_UPSTREAM_BASE_URL=http://127.0.0.1:8080/v1
set EVPRTR_HOST=127.0.0.1
set EVPRTR_PORT=8741
python -m compositor
```

## Build & Test

```bash
ruff check .
ruff format --check .
pytest
```

## Development Principles

- Prefer mechanism (routing, schemas, verify/repair, traces) over training or distillation.
- Evaluation captures failure locus for debugging; it is not a leaderboard project.
- Keep the OpenAI-compatible contract harness-safe (`tool_calls` shape, streaming when added).

## Architectural Boundaries

1. **Compositor calls runtimes; it does not embed inference engines.**
2. **Harness (upper) ≠ compositor (middle) ≠ runtime (lower).** Do not blur these names in docs or code.
3. Public HTTP surface stays OpenAI-compatible so existing harnesses can aim a base URL here.
4. Traces/observability are first-class for improvement feedback; score chasing is not.
5. **Verify policies are injectable.** Generic detectors/repair live under `compositor/verify/` with stable `policy_id`s; Maple-Preview–oriented heuristics are peelable modules (`detectors_maple_preview.py`, `MaplePreviewRepair`). Do not bury model-family heuristics inside `core.py`.
6. **Tool determination may use Needle 2** (`compositor/tools/`, `runtimes/needle.py`) as a peelable path. Needle must not execute harness tools; it only proposes structured calls / refusals. Maple remains the prose/reasoning runtime.
7. **Side-effect tool calls are buffered** (`compositor/approvals/`) for review before apply. Do not re-enable passthrough of `write`/`bash`/`web_search` without an explicit policy change.

## Prohibitions

1. Do not add distillation / SFT / weight-finetune pipelines unless the user explicitly revisits that policy.
2. Do not commit model weights, GGUF files, or secrets.
3. Do not name compositor modules `runtime` or `harness` in a way that collides with the layer vocabulary.
4. Do not break `/v1/chat/completions` response shapes that harnesses depend on without a migration note in `docs/`.

## Git Conventions

Scoped Conventional Commits when helpful: `feat(api):`, `feat(compositor):`, `docs:`.

## Session Handoff

Long-running workstreams use GitHub issues for cross-session continuity. See `docs/handoff-protocol.md` for the full protocol.

- Label: `session-handoff`
- One issue per workstream (not per session)
- On session start, read the relevant handoff issue and confirm the **Next action** with the user before executing.

## Internationalisation

Follow `docs/i18n-policy.md`:

- Translations are suffix files (`README.ja.md` next to `README.md`); no language directories.
- In scope: `README.md`, `docs/overview.md`. `docs/architecture.md` stays English-only.
- Each translated file carries a `> Source: <name>.md @ <sha>` header. PRs are never blocked on translation parity.

---

<!-- Common rules below this line apply to every project. -->

## Common Development Rules

### TDD (Red → Green → Refactor)

All implementation work proceeds in this cycle:

1. **Red**: write a failing test that captures the intended behaviour.
2. **Green**: write the minimum code that makes the test pass.
3. **Refactor**: tidy up while keeping tests green.

When a test fails, fix the production code — do not delete, skip, or weaken the test.

### Git Conventions

- **Conventional Commits**: `feat:` `fix:` `docs:` `refactor:` `test:` `ci:` `chore:`. Project-specific prefixes (e.g. `data:`, `experiments:`) live in the project's `AGENTS.md`.
- **Branch naming**: use a short prefix for the agent or author followed by a topic, e.g. `claude/<topic>`, `codex/<topic>`, or `human/<topic>`.
- **Trailer**: when an AI agent authors the commit, append a trailer crediting the agent. Do not embed model name or session info in the trailer; put those in the commit body if needed.
- **Pre-push hook**: install via `cp git-hooks/pre-push .git/hooks/pre-push && chmod +x .git/hooks/pre-push` (or `git config core.hooksPath git-hooks`). The hook runs format / lint / clippy before every push. Tests are intentionally omitted — TDD keeps them green at commit time.

### Pull Requests

- **Always ready for review.** Open PRs in the "ready" state, never as drafts. Draft PRs do not fire review-requested events and slow the loop.
- **Auto-subscribe after creating a PR.** Immediately after the PR is created, subscribe to its activity without asking the user. Rationale: the user explicitly opted into the "agent opens and watches its own PRs" workflow at the template level, so the per-PR confirmation is noise. Unsubscribe only when the user says to stop, when the PR merges, or when it is closed unmerged.
- **One PR per workstream**, matching the handoff issue. Reference the issue with `Closes #N` per `.github/PULL_REQUEST_TEMPLATE.md`.

### Stream Idle Timeout Mitigation

Cloud agent sessions occasionally fail with `Stream idle timeout - partial response received` on long output. To reduce risk:

1. **Stage long writes.** For long documents or source files, write the skeleton (headings, function signatures, trait stubs) first, then fill each section in follow-up edits. Avoid single blocks larger than ~200 lines.
2. **Watch out after large reads.** Reading a big file (e.g. `Cargo.lock`, large generated modules) and then immediately producing long output is a common trigger. Split into separate turns or excerpt only the relevant portion.
3. **Recover carefully.** A timeout can still leave the file write completed. Run `git status` before retrying so the same content is not written twice.

### Common Prohibitions

1. Do not delete, skip, or comment out existing tests.
2. Do not modify CI configuration without explicit instruction.
3. Do not weaken production code merely to make tests pass.
4. Do not commit credentials, API keys, signed URLs, or anything in `.env*`.
