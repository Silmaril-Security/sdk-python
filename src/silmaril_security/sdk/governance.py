# Copyright (c) 2024-2026 Silmaril Security Inc. All rights reserved.

"""Governance resource identity helpers."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from silmaril_security.sdk.types import GovernanceResource

McpIdentityResolutionStatus = Literal["resolved", "unresolved", "ambiguous"]
_MCP_TOOL_NAMES = (
    re.compile(r"^mcp__(.+?)__(.+)$"),
    re.compile(r"^MCP:([^:]+):(.+)$"),
)


@dataclass(frozen=True)
class McpIdentityResolution:
    """Explicit result of resolving a host MCP tool name."""

    status: McpIdentityResolutionStatus
    resource: GovernanceResource | None = None


def resolve_mcp_tool_identity(
    host_tool_name: str,
    configured_server_ids: Sequence[str],
) -> McpIdentityResolution:
    """Resolve a host MCP tool name using only supplied configured server IDs.

    Exact configured IDs win. Otherwise, a host alias formed by replacing
    hyphens with underscores must identify exactly one configured server.
    """

    parsed = _parse_host_tool_name(host_tool_name)
    if parsed is None:
        return McpIdentityResolution(status="unresolved")
    host_server_id, tool_id = parsed

    server_ids = tuple(
        dict.fromkeys(
            server_id
            for server_id in configured_server_ids
            if isinstance(server_id, str) and _has_non_whitespace(server_id)
        )
    )
    if host_server_id in server_ids:
        return McpIdentityResolution(
            status="resolved",
            resource=GovernanceResource(
                kind="mcp_tool",
                id=tool_id,
                parent_id=host_server_id,
            ),
        )

    candidates = [
        server_id
        for server_id in server_ids
        if server_id.replace("-", "_") == host_server_id
    ]
    if len(candidates) == 1:
        return McpIdentityResolution(
            status="resolved",
            resource=GovernanceResource(
                kind="mcp_tool",
                id=tool_id,
                parent_id=candidates[0],
            ),
        )
    return McpIdentityResolution(
        status="ambiguous" if len(candidates) > 1 else "unresolved"
    )


def _parse_host_tool_name(host_tool_name: object) -> tuple[str, str] | None:
    if not isinstance(host_tool_name, str):
        return None
    for pattern in _MCP_TOOL_NAMES:
        match = pattern.fullmatch(host_tool_name)
        if match is not None:
            server_id, tool_id = match.groups()
            if _has_non_whitespace(server_id) and _has_non_whitespace(tool_id):
                return server_id, tool_id
    return None


def _has_non_whitespace(value: str) -> bool:
    return bool(value) and any(not character.isspace() for character in value)
