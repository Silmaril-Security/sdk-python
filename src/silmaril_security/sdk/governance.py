# Copyright (c) 2024-2026 Silmaril Security Inc. All rights reserved.

"""Governance resource identity helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

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
    configured_server_ids: Sequence[str | Mapping[str, Any]],
    configured_tools: Sequence[Mapping[str, Any]] | None = None,
    *,
    authoritative_resource: GovernanceResource | Mapping[str, Any] | None = None,
) -> McpIdentityResolution:
    """Resolve one raw MCP dispatch name against a configured catalog.

    A supplied canonical resource is authoritative. Otherwise every configured
    server ID, explicit alias, and deterministic host alias is one candidate
    set: an exact ID does not outrank an alias. The host alias replaces
    hyphens in the configured server ID with underscores and never rewrites
    underscores as hyphens. Server-only catalogs take the nonempty remainder
    after a separator-bounded prefix, including further separators. A tool
    catalog matches complete ``mcp__{key}__{tool}`` or ``MCP:{key}:{tool}``
    spellings. Identical canonical refs count once. Several distinct refs are
    ambiguous, and none is unresolved.
    """

    if authoritative_resource is not None:
        resource = (
            authoritative_resource
            if isinstance(authoritative_resource, GovernanceResource)
            else GovernanceResource.from_wire(authoritative_resource)
        )
        return McpIdentityResolution(status="resolved", resource=resource)

    form = _host_form(host_tool_name)
    if form is None:
        return McpIdentityResolution(status="unresolved")
    marker, separator = form
    body = host_tool_name[len(marker) :]
    servers = _server_rows(configured_server_ids)
    if configured_tools is None:
        candidates = _server_prefix_candidates(body, servers, separator)
    else:
        candidates = _tool_spelling_candidates(
            body, servers, configured_tools, separator
        )
    unique = tuple(dict.fromkeys(candidates))
    if len(unique) == 1:
        return _resolved(*unique[0])
    if len(unique) > 1:
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


def _server_rows(
    configured_server_ids: Sequence[str | Mapping[str, Any]],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    rows: list[tuple[str, tuple[str, ...]]] = []
    for entry in configured_server_ids:
        if isinstance(entry, str):
            server_id, aliases = entry, ()
        elif isinstance(entry, Mapping) and isinstance(entry.get("id"), str):
            server_id = entry["id"]
            raw_aliases = entry.get("aliases") or ()
            aliases = tuple(
                alias
                for alias in raw_aliases
                if isinstance(alias, str) and alias != server_id
            )
        else:
            continue
        if not _valid_id(server_id):
            continue
        keys = [server_id, *[alias for alias in aliases if _valid_id(alias)]]
        host_alias = server_id.replace("-", "_")
        if host_alias != server_id and _valid_id(host_alias):
            keys.append(host_alias)
        rows.append((server_id, tuple(dict.fromkeys(keys))))
    return tuple(rows)


def _server_prefix_candidates(
    body: str,
    servers: Sequence[tuple[str, tuple[str, ...]]],
    separator: str,
) -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []
    for server_id, keys in servers:
        for key in keys:
            prefix = f"{key}{separator}"
            if not body.startswith(prefix):
                continue
            tool_id = body[len(prefix) :]
            if _valid_id(tool_id):
                candidates.append((server_id, tool_id))
    return candidates


def _tool_spelling_candidates(
    body: str,
    servers: Sequence[tuple[str, tuple[str, ...]]],
    configured_tools: Sequence[Mapping[str, Any]],
    separator: str,
) -> list[tuple[str, str]]:
    keys_by_server = {server_id: keys for server_id, keys in servers}
    candidates: list[tuple[str, str]] = []
    for entry in configured_tools:
        if not isinstance(entry, Mapping):
            continue
        tool_id = entry.get("id")
        parent_id = entry.get("parent_id")
        if not isinstance(tool_id, str) or not isinstance(parent_id, str):
            continue
        if not _valid_id(tool_id) or not _valid_id(parent_id):
            continue
        for key in keys_by_server.get(parent_id, (parent_id,)):
            if body == f"{key}{separator}{tool_id}":
                candidates.append((parent_id, tool_id))
    return candidates


def _resolved(server_id: str, tool_id: str) -> McpIdentityResolution:
    return McpIdentityResolution(
        status="resolved",
        resource=GovernanceResource(
            kind="mcp_tool",
            id=tool_id,
            parent_id=server_id,
        ),
    )


def _valid_id(value: str) -> bool:
    return (
        bool(value)
        and any(not character.isspace() for character in value)
        and "\n" not in value
        and "\r" not in value
    )
