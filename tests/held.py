# Copyright (c) 2024-2026 Silmaril Security Inc. All rights reserved.

"""Classifications a test holds open so the host flow can be observed beside them."""

from __future__ import annotations

import asyncio
import threading
from typing import Any

from silmaril_security.sdk import BlockResult

WAIT_SECONDS = 10


def verdict(prediction: str = "BENIGN", mode: str | None = "block") -> BlockResult:
    return BlockResult(prediction, 0.9 if prediction == "MALICIOUS" else 0.1, 0.5, mode=mode)


class HeldClassification:
    """Replace ``Firewall._classify_raw``; each call waits until ``release`` is set."""

    def __init__(self, *predictions: str) -> None:
        self.predictions = list(predictions) or ["BENIGN"]
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.started = threading.Event()
        self.release = threading.Event()

    def __call__(self, text: str, **options: Any) -> BlockResult:
        self.calls.append((text, options))
        self.started.set()
        assert self.release.wait(WAIT_SECONDS), "held classification was never released"
        return verdict(self.predictions[min(len(self.calls), len(self.predictions)) - 1], options.get("mode"))

    @property
    def hooks(self) -> list[Any]:
        return [options["hook"] for _, options in self.calls]


class AsyncHeldClassification:
    """Replace ``AsyncFirewall._classify_raw``; each call waits until ``release`` is set."""

    def __init__(self, *predictions: str) -> None:
        self.predictions = list(predictions) or ["BENIGN"]
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def __call__(self, text: str, **options: Any) -> BlockResult:
        self.calls.append((text, options))
        self.started.set()
        await asyncio.wait_for(self.release.wait(), WAIT_SECONDS)
        return verdict(self.predictions[min(len(self.calls), len(self.predictions)) - 1], options.get("mode"))

    @property
    def hooks(self) -> list[Any]:
        return [options["hook"] for _, options in self.calls]
