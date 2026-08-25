"""Detect runaway token/phrase repetition in generative text.

Generic: not tied to a specific model family. Used by the default verify stack
and reusable when peeling Maple-specific heuristics later.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class RepetitionHit:
    kind: str
    onset: int
    detail: dict[str, str | int]


_WORD_RE = re.compile(r"\S+")
# Short motif repeated many times: "agent agent…", "trtrtr…", "composite composite…"
_CHAR_MOTIF_RE = re.compile(r"(.{2,24}?)\1{7,}")


def find_repetition(
    text: str,
    *,
    word_run: int = 8,
    ngram_size: int = 3,
    ngram_run: int = 5,
) -> RepetitionHit | None:
    """Return the earliest collapse onset, or None if text looks healthy."""
    if not text or len(text) < 32:
        return None

    hits: list[RepetitionHit] = []

    words = list(_WORD_RE.finditer(text))
    if len(words) >= word_run:
        run = 1
        for i in range(1, len(words)):
            if words[i].group() == words[i - 1].group():
                run += 1
                if run >= word_run:
                    onset = words[i - run + 1].start()
                    hits.append(
                        RepetitionHit(
                            kind="word_run",
                            onset=onset,
                            detail={"token": words[i].group()[:80], "run": run},
                        )
                    )
                    break
            else:
                run = 1

    if len(words) >= ngram_size * ngram_run:
        grams = [
            " ".join(words[i + j].group() for j in range(ngram_size))
            for i in range(len(words) - ngram_size + 1)
        ]
        run = 1
        for i in range(1, len(grams)):
            if grams[i] == grams[i - 1]:
                run += 1
                if run >= ngram_run:
                    onset = words[i - run + 1].start()
                    hits.append(
                        RepetitionHit(
                            kind="ngram_run",
                            onset=onset,
                            detail={"ngram": grams[i][:120], "run": run, "n": ngram_size},
                        )
                    )
                    break
            else:
                run = 1

    match = _CHAR_MOTIF_RE.search(text)
    if match:
        hits.append(
            RepetitionHit(
                kind="char_motif",
                onset=match.start(),
                detail={"motif": match.group(1)[:80], "span": match.end() - match.start()},
            )
        )

    if not hits:
        return None
    return min(hits, key=lambda h: h.onset)


def truncate_before_repetition(text: str, hit: RepetitionHit | None = None) -> str:
    hit = hit or find_repetition(text)
    if hit is None:
        return text
    return text[: hit.onset].rstrip()


def message_fields(message: dict) -> dict[str, str]:
    """Extract text-bearing fields from an assistant message."""
    out: dict[str, str] = {}
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        out["content"] = content
    reasoning = message.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning.strip():
        out["reasoning_content"] = reasoning
    return out
