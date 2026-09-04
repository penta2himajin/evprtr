"""Needle 2 query/tool contract helpers (aligned with cactus-needle apis.md).

Official model: short natural-language queries, ≤5 tools rendered (else silent
retrieval), optional ``system`` facts (not instructions), ~256 decode budget,
confidence gating.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from compositor.tools.maple_nl import needle_retry_instruction

# Prefer coding-agent tools when truncating catalogues for Needle's 5-tool slot.
_TOOL_PRIORITY: tuple[str, ...] = (
    "write",
    "edit",
    "read",
    "ls",
    "grep",
    "find",
    "bash",
    "powershell",
    "glob",
    "list_dir",
)

DEFAULT_NEEDLE_MAX_TOOLS = 5
DEFAULT_NEEDLE_QUERY_LIMIT = 160


def select_needle_tools(
    tools: list[dict[str, Any]],
    *,
    hint: str = "",
    max_tools: int = DEFAULT_NEEDLE_MAX_TOOLS,
) -> list[dict[str, Any]]:
    """Pick at most ``max_tools`` schemas so Needle does not silently retrieve."""
    if max_tools < 1:
        return []
    if len(tools) <= max_tools:
        return list(tools)

    hint_l = (hint or "").lower()
    scored: list[tuple[float, int, dict[str, Any]]] = []
    for idx, tool in enumerate(tools):
        name = str(tool.get("name") or "")
        score = 0.0
        if name and name.lower() in hint_l:
            score += 100.0
        desc = str(tool.get("description") or "").lower()
        if name and any(tok and tok in hint_l for tok in name.lower().replace("_", " ").split()):
            score += 20.0
        if any(w and w in hint_l for w in desc.split()[:8]):
            score += 5.0
        if name in _TOOL_PRIORITY:
            score += float(40 - _TOOL_PRIORITY.index(name))
        score -= idx * 0.01
        scored.append((score, idx, tool))

    scored.sort(key=lambda row: (-row[0], row[1]))
    picked = [row[2] for row in scored[:max_tools]]

    # Stable presentation order: priority list, then original order.
    def _order_key(tool: dict[str, Any]) -> tuple[int, int]:
        name = str(tool.get("name") or "")
        if name in _TOOL_PRIORITY:
            return (0, _TOOL_PRIORITY.index(name))
        return (1, next(i for i, t in enumerate(tools) if t is tool or t.get("name") == name))

    return sorted(picked, key=_order_key)


def normalize_needle_query(
    text: str,
    *,
    user_task: str,
    tool_names: list[str],
    limit: int = DEFAULT_NEEDLE_QUERY_LIMIT,
) -> str:
    """Force short imperative queries (Needle's 256-token window + training shape)."""
    raw = (text or "").strip()
    if not raw:
        return needle_retry_instruction(
            maple_nl=user_task,
            user_task=user_task,
            tool_names=tool_names,
            limit=limit,
        )
    compact = " ".join(raw.split())
    if len(compact) <= limit and compact.lower().startswith("call "):
        return compact[:limit]
    return needle_retry_instruction(
        maple_nl=raw,
        user_task=user_task,
        tool_names=tool_names,
        limit=limit,
    )


def needle_system_facts(*, cwd: str | None = None) -> str:
    """Environment facts only (apis.md: instructions in ``system`` do not steer)."""
    now = datetime.now().strftime("%Y-%m-%d %a %H:%M")
    parts = [
        f"date: {now}",
        "locale: en-US",
        "device: coding-agent",
    ]
    if cwd:
        parts.append(f"location: {cwd}")
    return "; ".join(parts)
