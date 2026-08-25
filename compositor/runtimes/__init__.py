"""Clients that call underlying model runtimes (not inference engines themselves)."""

from __future__ import annotations

from typing import Any, Protocol


class ChatRuntime(Protocol):
    """A downstream runtime that speaks OpenAI-compatible chat completions."""

    async def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Forward a chat.completions body; return the upstream JSON object."""


class UpstreamError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None, body: str | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body
