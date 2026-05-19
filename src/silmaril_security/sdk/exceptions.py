# Copyright (c) 2024-2026 Silmaril Security Inc. All rights reserved.

"""Exceptions raised by the Silmaril Firewall SDK."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from typing_extensions import deprecated

from silmaril_security.sdk.hooks import HookLabel
from silmaril_security.sdk.types import BlockedBatchItem, BlockResult

_MAX_PROMPT_DISPLAY_LEN = 100


class SilmarilApiError(Exception):
    """Raised when the firewall API returns a non-success HTTP status."""

    def __init__(self, *, status: int, status_text: str, body: str) -> None:
        self.status = status
        self.status_text = status_text
        self.body = body
        super().__init__(f"Silmaril API error {status} {status_text}".rstrip())


APIError = SilmarilApiError


class FirewallBlockedException(Exception):
    """Raised when a single classification is blocked by the Firewall."""

    def __init__(
        self,
        *,
        score: float,
        threshold: float,
        prompt_text: str,
        hook: HookLabel = HookLabel.UNKNOWN,
        tool_name: str | None = None,
        result: BlockResult | None = None,
        run_id: UUID | str | None = None,
    ) -> None:
        self.score = score
        self.threshold = threshold
        self.prompt_text = prompt_text
        self.hook = hook
        self.tool_name = tool_name
        self.result = result
        self.run_id = run_id
        super().__init__(str(self))

    def __str__(self) -> str:
        truncated = self.prompt_text
        if len(truncated) > _MAX_PROMPT_DISPLAY_LEN:
            truncated = truncated[:_MAX_PROMPT_DISPLAY_LEN] + "..."
        return (
            "Request blocked by Silmaril Firewall "
            f"(score={self.score:.4f}, threshold={self.threshold:.4f}): "
            f"{truncated!r}"
        )


class BatchFirewallBlockedException(Exception):
    """Raised when one or more batch items are blocked by the Firewall."""

    def __init__(
        self,
        *,
        blocked: list[BlockedBatchItem],
        results: list[BlockResult],
    ) -> None:
        self.blocked = blocked
        self.results = results
        super().__init__(str(self))

    def __str__(self) -> str:
        if len(self.blocked) == 1:
            item = self.blocked[0]
            return (
                "Batch request blocked by Silmaril Firewall "
                f"at index {item.index} "
                f"(score={item.result.score:.4f}, threshold={item.result.threshold:.4f})"
            )
        return f"Batch requests blocked by Silmaril Firewall ({len(self.blocked)} items)"


if TYPE_CHECKING:

    @deprecated("Use FirewallBlockedException instead.")
    class PromptBlockedException(FirewallBlockedException):
        pass

    @deprecated("Use BatchFirewallBlockedException instead.")
    class BatchPromptBlockedException(BatchFirewallBlockedException):
        pass

else:
    PromptBlockedException = FirewallBlockedException
    BatchPromptBlockedException = BatchFirewallBlockedException
