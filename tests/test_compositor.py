import pytest

from compositor.core import Compositor


class _Runtime:
    async def chat_completions(self, payload):
        return {
            "model": "raw-upstream",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}}],
        }


@pytest.mark.asyncio
async def test_compositor_sets_public_model_and_defaults():
    c = Compositor(_Runtime(), public_model_id="evprtr")
    result = await c.chat_completions({"messages": []})
    out = result.response
    assert out["model"] == "evprtr"
    assert out["object"] == "chat.completion"
    assert out["id"]
    assert out["created"] > 0
    assert result.trace.ok is True
