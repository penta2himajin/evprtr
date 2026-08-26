import json

from compositor.runtimes.needle import NeedleCompleteResult, NeedleToolRuntime
from compositor.tools.convert import (
    filter_tools_for_choice,
    needle_calls_to_openai_message,
    openai_tools_to_needle,
    should_route_tools_to_needle,
)
from compositor.tools.path import NeedleToolPath


OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["city"],
            },
        },
    }
]


def test_openai_tools_to_needle():
    needle = openai_tools_to_needle(OPENAI_TOOLS)
    assert needle == [
        {
            "name": "get_weather",
            "description": "Get weather",
            "parameters": OPENAI_TOOLS[0]["function"]["parameters"],
        }
    ]


def test_should_route_skips_tool_choice_none():
    assert (
        should_route_tools_to_needle({"tools": OPENAI_TOOLS, "tool_choice": "none"}, enabled=True)
        is False
    )
    assert (
        should_route_tools_to_needle({"tools": OPENAI_TOOLS, "tool_choice": "auto"}, enabled=True)
        is True
    )


def test_filter_forced_tool():
    tools = openai_tools_to_needle(
        OPENAI_TOOLS
        + [
            {
                "type": "function",
                "function": {
                    "name": "get_time",
                    "description": "time",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
    )
    filtered = filter_tools_for_choice(
        tools, {"type": "function", "function": {"name": "get_weather"}}
    )
    assert [t["name"] for t in filtered] == ["get_weather"]


def test_needle_calls_to_openai_message_args_json():
    msg = needle_calls_to_openai_message(
        [{"name": "get_weather", "arguments": {"city": "Tokyo", "unit": "celsius"}}]
    )
    assert msg["tool_calls"][0]["function"]["name"] == "get_weather"
    assert json.loads(msg["tool_calls"][0]["function"]["arguments"])["city"] == "Tokyo"


def test_tool_path_returns_structured_tool_calls():
    class RT(NeedleToolRuntime):
        def available(self):
            return True

        def complete(self, query, tools):
            return NeedleCompleteResult(
                function_calls=[
                    {
                        "name": "get_weather",
                        "arguments": {"city": "Tokyo", "unit": "celsius"},
                    }
                ],
                confidence=0.9,
                reasoning=None,
                raw={},
            )

    path = NeedleToolPath(RT())
    result = path.handle(
        {
            "messages": [{"role": "user", "content": "Weather in Tokyo celsius?"}],
            "tools": OPENAI_TOOLS,
            "tool_choice": "auto",
        }
    )
    assert result is not None
    assert result.empty_call is False
    msg = result.response["choices"][0]["message"]
    assert msg["tool_calls"][0]["function"]["name"] == "get_weather"
    assert result.response["choices"][0]["finish_reason"] == "tool_calls"


def test_tool_path_schema_conflict_required_refuses_without_markup():
    class RT(NeedleToolRuntime):
        def available(self):
            return True

        def complete(self, query, tools):
            return NeedleCompleteResult(
                function_calls=[],
                confidence=0.95,
                reasoning="city is required; cannot omit",
                raw={},
            )

    path = NeedleToolPath(RT())
    result = path.handle(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Call get_weather with unit=celsius but omit city",
                }
            ],
            "tools": OPENAI_TOOLS,
            "tool_choice": "required",
        }
    )
    assert result is not None
    assert result.empty_call is True
    content = result.response["choices"][0]["message"]["content"]
    assert "<tool_call" not in content.lower()
    assert "schema-valid" in content.lower() or "required" in content.lower()
    assert result.response["choices"][0]["message"].get("tool_calls") in (None, [])


def test_tool_path_skip_reason_empty_converted_tools():
    class RT(NeedleToolRuntime):
        def available(self):
            return True

        def complete(self, query, tools):
            raise AssertionError("should not call Needle")

    path = NeedleToolPath(RT())
    result = path.handle(
        {
            "messages": [{"role": "user", "content": "do something useful now"}],
            "tools": [{"type": "function", "function": {}}],  # no name
            "tool_choice": "auto",
        }
    )
    assert result is None
    assert path.last_skip_reason is not None
    assert "empty_converted_tools" in path.last_skip_reason


def test_tool_path_skip_reason_empty_query():
    class RT(NeedleToolRuntime):
        def available(self):
            return True

        def complete(self, query, tools):
            raise AssertionError("should not call Needle")

    path = NeedleToolPath(RT())
    result = path.handle(
        {
            "messages": [{"role": "system", "content": "only system"}],
            "tools": OPENAI_TOOLS,
            "tool_choice": "auto",
        }
    )
    assert result is None
    assert path.last_skip_reason == "empty_query"


def test_last_user_text_skips_stub_continue():
    from compositor.tools.convert import last_user_text

    text = last_user_text(
        [
            {"role": "user", "content": "Implement the HTTP server with axum on port 3000."},
            {"role": "user", "content": "continue"},
        ]
    )
    assert "Implement the HTTP server" in text
    assert text != "continue"


def test_tool_path_empty_call_auto_falls_back_to_maple():
    class RT(NeedleToolRuntime):
        def available(self):
            return True

        def complete(self, query, tools):
            return NeedleCompleteResult(
                function_calls=[],
                confidence=0.03,
                reasoning="No tool exists for creating files",
                raw={},
            )

    path = NeedleToolPath(RT())
    result = path.handle(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Create rpc-smoke.txt with write tool only.",
                }
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "write",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"},
                                "content": {"type": "string"},
                            },
                            "required": ["path", "content"],
                        },
                    },
                }
            ],
            "tool_choice": "auto",
        }
    )
    assert result is None
    assert path.last_skip_reason is not None
    assert "empty_call_fallback" in path.last_skip_reason

