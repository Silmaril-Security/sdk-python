# Copyright (c) 2024-2026 Silmaril Security Inc. All rights reserved.

"""Text sanitization shared by sync and async clients."""

from __future__ import annotations


def sanitize_text(text: str) -> str:
    """Remove unpaired Unicode surrogates without splitting the event."""
    return text.encode("utf-8", errors="ignore").decode("utf-8")
