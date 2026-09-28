# Copyright (c) 2024-2026 Silmaril Security Inc. All rights reserved.

"""Governance resource identity helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from silmaril_security.sdk.types import GovernanceResource

McpIdentityResolutionStatus = Literal["resolved", "unresolved", "ambiguous"]
_HOST_FORMS = (
    ("mcp__", "__"),
    ("MCP:", ":"),
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

    Exact configured IDs are matched as complete prefixes, so an ID may itself
    contain ``__`` or ``:``. When several exact prefixes match, the separator
    boundary that extends furthest wins. Equal boundaries stay ambiguous. A
    hyphen-to-underscore alias is considered only when no exact prefix matches,
    and only a unique alias resolves.
    """

    form = _host_form(host_tool_name)
    if form is None:
        return McpIdentityResolution(status="unresolved")
    marker, separator = form
    server_ids = _configured_server_ids(configured_server_ids)

    exact = _prefix_matches(
        host_tool_name,
        server_ids,
        marker,
        separator,
        alias=False,
    )
    selected = _unique_furthest_boundary(exact)
    if selected is not None:
        return _resolved(*selected)
    if exact:
        return McpIdentityResolution(status="ambiguous")

    aliases = _prefix_matches(
        host_tool_name,
        server_ids,
        marker,
        separator,
        alias=True,
    )
    if len(aliases) == 1:
        server_id, tool_id, _boundary = aliases[0]
        return _resolved(server_id, tool_id)
    if len(aliases) > 1:
        return McpIdentityResolution(status="ambiguous")
    return McpIdentityResolution(status="unresolved")


def _host_form(host_tool_name: object) -> tuple[str, str] | None:
    if (
        not isinstance(host_tool_name, str)
        or "\n" in host_tool_name
        or "\r" in host_tool_name
    ):
        return None
    for marker, separator in _HOST_FORMS:
        if host_tool_name.startswith(marker):
            return marker, separator
    return None


def _configured_server_ids(server_ids: Sequence[str]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            server_id
            for server_id in server_ids
            if isinstance(server_id, str)
            and _has_non_whitespace(server_id)
            and "\n" not in server_id
            and "\r" not in server_id
        )
    )


def _prefix_matches(
    host_tool_name: str,
    server_ids: Sequence[str],
    marker: str,
    separator: str,
    *,
    alias: bool,
) -> list[tuple[str, str, int]]:
    matches: list[tuple[str, str, int]] = []
    for server_id in server_ids:
        observed_id = server_id.replace("-", "_") if alias else server_id
        prefix = f"{marker}{observed_id}{separator}"
        if not host_tool_name.startswith(prefix):
            continue
        tool_id = host_tool_name[len(prefix) :]
        if _has_non_whitespace(tool_id):
            matches.append((server_id, tool_id, len(prefix)))
    return matches


def _unique_furthest_boundary(
    matches: Sequence[tuple[str, str, int]],
) -> tuple[str, str] | None:
    if not matches:
        return None
    furthest = max(boundary for _server_id, _tool_id, boundary in matches)
    winners = [match for match in matches if match[2] == furthest]
    if len(winners) != 1:
        return None
    server_id, tool_id, _boundary = winners[0]
    return server_id, tool_id


def _resolved(server_id: str, tool_id: str) -> McpIdentityResolution:
    return McpIdentityResolution(
        status="resolved",
        resource=GovernanceResource(
            kind="mcp_tool",
            id=tool_id,
            parent_id=server_id,
        ),
    )


def _has_non_whitespace(value: str) -> bool:
    return bool(value) and any(not character.isspace() for character in value)
