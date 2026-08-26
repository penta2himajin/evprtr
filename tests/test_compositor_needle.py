from pathlib import Path

import pytest

from compositor.core import Compositor
from compositor.runtimes.needle import NeedleCompleteResult, NeedleToolRuntime
from compositor.tools.path import NeedleToolPath
from compositor.trace import TraceStore


class _MapleUnused:
    async def chat_completions(self, payload):
        raise AssertionError("Maple should not be called for tool path")


@pytest.mark.asyncio
async def test_compositor_uses_needle_for_tools(tmp_path: Path):
    class RT(NeedleToolRuntime):
        def available(self):
            return True

        def complete(self, query, tools):
            return NeedleCompleteResult(
                function_calls=[{"name": "get_weather", "arguments": {"city": "Tokyo"}}],
                confidence=0.88,
                reasoning=None,
                raw={},
            )

    c = Compositor(
        _MapleUnused(),
        traces=TraceStore(tmp_path),
        tool_path=NeedleToolPath(RT()),
        needle_via_maple_nl=False,
        maple_tools_primary=False,
        needle_correct_degenerate=False,
    )
    result = await c.chat_completions(
        {
            "messages": [{"role": "user", "content": "Weather in Tokyo?"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                }
            ],
            "tool_choice": "auto",
        }
    )
    assert result.trace.response_summary["needle_tool_path"] is True
    assert result.response["choices"][0]["finish_reason"] == "tool_calls"
    assert any(e.stage == "tool_select" for e in result.trace.events)
