# Copyright (c) 2024-2026 Silmaril Security Inc. All rights reserved.

"""Client-side text chunking for long firewall inputs."""

from __future__ import annotations

CHARS_PER_TOKEN = 4

MAX_INPUT_TOKENS = 10_240
CHUNK_WINDOW = 400
CHUNK_OVERLAP = 64

MAX_INPUT_CHARS = MAX_INPUT_TOKENS * CHARS_PER_TOKEN
CHUNK_WINDOW_CHARS = CHUNK_WINDOW * CHARS_PER_TOKEN
CHUNK_OVERLAP_CHARS = CHUNK_OVERLAP * CHARS_PER_TOKEN
CHUNK_STRIDE_CHARS = CHUNK_WINDOW_CHARS - CHUNK_OVERLAP_CHARS


def chunk_text(text: str) -> list[str]:
    """Split text into overlapping windows.

    Short inputs are returned as a single-element list. Long inputs are split
    into 400-token windows with a 64-token overlap, using the SDK-wide
    4-characters-per-token approximation.
    """
    n = len(text)
    if n > MAX_INPUT_CHARS:
        raise ValueError(
            f"firewall: input has ~{n // CHARS_PER_TOKEN} tokens ({n} chars); "
            f"max is {MAX_INPUT_TOKENS} tokens ({MAX_INPUT_CHARS} chars)"
        )
    if n <= CHUNK_WINDOW_CHARS:
        return [text]

    chunks: list[str] = []
    for start in range(0, n, CHUNK_STRIDE_CHARS):
        end = start + CHUNK_WINDOW_CHARS
        chunks.append(text[start:end])
        if end >= n:
            break
    return chunks
