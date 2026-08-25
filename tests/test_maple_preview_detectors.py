import json
from pathlib import Path

import pytest

from compositor.core import Compositor
from compositor.trace import TraceStore
from compositor.verify.detectors_maple_preview import (
    EmptyContentLongReasoningDetector,
    PseudoToolCallInContentDetector,
    ThinContentDetector,
    strip_pseudo_tool_markup,
)
from compositor.verify.pipeline import VerifyBundle

FIXTURE_PSEUDO = Path("tests/fixtures/maple_tool_failures/pseudo_tool_choice_none.json")


def test_thin_content_detector_flags_preamble():
    d = ThinContentDetector().diagnose({"role": "assistant", "content": "Here are five bullets:"})
    assert d is not None
    assert d.policy_id == "maple_preview.thin_content"
    assert d.kind == "thin_content"


def test_thin_content_detector_allows_delivered_bullets():
    d = ThinContentDetector().diagnose({"role": "assistant", "content": "Points:\n- a\n- b"})
    assert d is None


def test_empty_long_reasoning_detector():
    d = EmptyContentLongReasoningDetector(min_reasoning_len=50).diagnose(
        {"role": "assistant", "content": "", "reasoning_content": "x" * 80}
    )
    assert d is not None
    assert d.policy_id == "maple_preview.empty_content_long_reasoning"


def test_pseudo_tool_markup_from_probe_fixture():
    data = json.loads(FIXTURE_PSEUDO.read_text(encoding="utf-8"))
    message = data["choices"][0]["message"]
    d = PseudoToolCallInContentDetector().diagnose(message)
    assert d is not None
    assert d.kind == "pseudo_tool_markup"
    cleaned = strip_pseudo_tool_markup(message["content"])
    assert "<tool_call" not in cleaned.lower()
    assert "I'll get the current weather" in cleaned


@pytest.mark.asyncio
async def test_maple_bundle_repairs_pseudo_tool_markup(tmp_path):
    bad = json.loads(FIXTURE_PSEUDO.read_text(encoding="utf-8"))
    good = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": (
                        "I cannot fetch live weather without tools. "
                        "Enable tool calling or check a weather service."
                    ),
                },
                "finish_reason": "stop",
            }
        ]
    }

    class Scripted:
        def __init__(self):
            self.calls = []
            self.responses = [bad, good]

        async def chat_completions(self, payload):
            self.calls.append(payload)
            return self.responses.pop(0)

    runtime = Scripted()
    c = Compositor(
        runtime,
        traces=TraceStore(tmp_path),
        verify=VerifyBundle.maple_preview(repair_attempts=1),
    )
    result = await c.chat_completions(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "What is the weather in Tokyo right now? Use live data.",
                }
            ],
            "tools": [{"type": "function", "function": {"name": "get_weather"}}],
            "tool_choice": "none",
        }
    )
    content = result.response["choices"][0]["message"]["content"] or ""
    assert "<tool_call" not in content.lower()
    assert "cannot fetch live weather" in content
    assert len(runtime.calls) == 2
    assert runtime.calls[1].get("tool_choice") == "none"
