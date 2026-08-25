"""Tool-path helpers (Needle integration)."""

from compositor.tools.convert import (
    filter_tools_for_choice,
    last_user_text,
    needle_calls_to_openai_message,
    openai_tools_to_needle,
    should_route_tools_to_needle,
)
from compositor.tools.path import NeedleToolPath, ToolPathResult

__all__ = [
    "NeedleToolPath",
    "ToolPathResult",
    "filter_tools_for_choice",
    "last_user_text",
    "needle_calls_to_openai_message",
    "openai_tools_to_needle",
    "should_route_tools_to_needle",
]
