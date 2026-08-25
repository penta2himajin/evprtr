"""HTTP client for an OpenAI-compatible chat runtime (e.g. llama-server)."""

from __future__ import annotations

from typing import Any

import httpx

from compositor.runtimes import UpstreamError


class OpenAICompatRuntime:
    def __init__(
        self,
        base_url: str,
        *,
        upstream_model: str | None = None,
        timeout: float = 600.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.upstream_model = upstream_model
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=timeout)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = dict(payload)
        if self.upstream_model:
            body["model"] = self.upstream_model
        # Compositor handles composition; streaming is not forwarded in v0.
        body["stream"] = False

        url = f"{self.base_url}/chat/completions"
        try:
            response = await self._client.post(url, json=body)
        except httpx.HTTPError as exc:
            raise UpstreamError(f"upstream request failed: {exc}") from exc

        if response.status_code >= 400:
            raise UpstreamError(
                f"upstream returned HTTP {response.status_code}",
                status_code=response.status_code,
                body=response.text,
            )
        return response.json()
