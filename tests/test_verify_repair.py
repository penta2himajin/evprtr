from __future__ import annotations

from pathlib import Path

import pytest

from compositor.core import Compositor
from compositor.trace import TraceStore
from compositor.verify.pipeline import VerifyBundle


class ScriptedRuntime:
    def __init__(self, responses: list[dict]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    async def chat_completions(self, payload):
        self.calls.append(payload)
        if not self.responses:
            raise AssertionError("no scripted responses left")
        return self.responses.pop(0)


def _assistant(content: str | None, reasoning: str | None = None) -> dict:
    message = {"role": "assistant", "content": content}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    return {
        "model": "maple",
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {},
    }


@pytest.mark.asyncio
async def test_truncates_collapsed_reasoning_when_content_ok(tmp_path: Path):
    collapsed = "plan: " + ("agent " * 30)
    runtime = ScriptedRuntime([_assistant("evprtr is a composite layer.", collapsed)])
    c = Compositor(runtime, traces=TraceStore(tmp_path), repair_attempts=0)
    result = await c.chat_completions({"messages": [{"role": "user", "content": "summarize"}]})
    message = result.response["choices"][0]["message"]
    assert message["content"] == "evprtr is a composite layer."
    assert "agent agent agent" not in (message.get("reasoning_content") or "")
    assert result.trace.response_summary["repetition_repaired"] is True
    assert any(e.stage == "verify" for e in result.trace.events)


@pytest.mark.asyncio
async def test_repairs_when_content_empty_and_reasoning_collapses(tmp_path: Path):
    bad = _assistant("", "The project is " + ("composite " * 40))
    good = _assistant(
        "- composite layer\n- called by harnesses\n- calls runtimes\n"
        "- no distillation\n- named after maple evaporator"
    )
    runtime = ScriptedRuntime([bad, good])
    c = Compositor(runtime, traces=TraceStore(tmp_path), repair_attempts=1)
    result = await c.chat_completions(
        {"messages": [{"role": "user", "content": "summarize the system"}]}
    )
    assert "composite layer" in result.response["choices"][0]["message"]["content"]
    assert len(runtime.calls) == 2
    repair_req = runtime.calls[1]
    assert repair_req["repeat_penalty"] >= 1.25
    assert repair_req["max_tokens"] <= 512
    assert result.trace.response_summary["repetition_repaired"] is True
    assert any(e.stage == "repair" for e in result.trace.events)
    assert any(
        e.detail.get("policy_id", "").startswith("maple_preview.")
        or e.detail.get("policy_id") == "generic.repetition"
        for e in result.trace.events
        if e.stage == "verify" and e.status == "error"
    )


@pytest.mark.asyncio
async def test_repairs_thin_preamble_content(tmp_path: Path):
    thin = _assistant("Here are the key points in exactly five bullet points:")
    good = _assistant("- one\n- two\n- three\n- four\n- five")
    runtime = ScriptedRuntime([thin, good])
    c = Compositor(runtime, traces=TraceStore(tmp_path), repair_attempts=1)
    result = await c.chat_completions(
        {"messages": [{"role": "user", "content": "five bullets please"}]}
    )
    assert "- one" in result.response["choices"][0]["message"]["content"]
    assert len(runtime.calls) == 2


@pytest.mark.asyncio
async def test_generic_only_stack_ignores_thin_content(tmp_path: Path):
    """Peeling Maple heuristics: thin preamble must not trigger repair."""
    thin = _assistant("Here are the key points in exactly five bullet points:")
    runtime = ScriptedRuntime([thin])
    c = Compositor(
        runtime,
        traces=TraceStore(tmp_path),
        verify=VerifyBundle.generic_only(repair_attempts=1),
    )
    result = await c.chat_completions(
        {"messages": [{"role": "user", "content": "five bullets please"}]}
    )
    assert len(runtime.calls) == 1
    assert result.trace.response_summary["repetition_repaired"] is False
    assert result.response["choices"][0]["message"]["content"].startswith("Here are")
