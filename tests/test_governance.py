# Copyright (c) 2024-2026 Silmaril Security Inc. All rights reserved.

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
import pytest

from silmaril_security.sdk import (
    AsyncFirewall,
    BatchFirewallBlockedException,
    BlockResult,
    Firewall,
    FirewallBlockedException,
    GovernanceDecision,
    GovernanceResource,
    resolve_mcp_tool_identity,
)

TEST_API_URL = "https://api.test.invalid/classify"
CONTRACT_DIR = Path(__file__).parents[1] / "contracts" / "governance" / "v1"


def governance_block() -> dict[str, Any]:
    return {
        "action": "block",
        "rule_id": "deny-github",
        "policy_version": "policy-7",
        "resource": {
            "kind": "mcp_tool",
            "id": "create_issue",
            "parent_id": "github",
        },
        "identity_revision": "catalog-9",
        "reason": "identity_unresolved",
    }


def test_vendored_contract_digest_and_concrete_resource_vectors():
    expected = {}
    for line in (CONTRACT_DIR / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        expected[name] = digest
    for name, digest in expected.items():
        assert hashlib.sha256((CONTRACT_DIR / name).read_bytes()).hexdigest() == digest

    vectors = json.loads((CONTRACT_DIR / "matching.json").read_text())
    resources = {
        (
            actual["kind"],
            actual["id"],
            actual.get("parent_id"),
        )
        for case in vectors["cases"]
        if (actual := case["actual"]) is not None
    }
    decoded = {
        (
            resource.kind,
            resource.id,
            resource.parent_id,
        )
        for resource in (
            GovernanceResource.from_wire(
                {
                    "kind": kind,
                    "id": resource_id,
                    **({"parent_id": parent_id} if parent_id is not None else {}),
                }
            )
            for kind, resource_id, parent_id in resources
        )
    }
    assert decoded == resources
    assert {kind for kind, _, _ in decoded} == {
        "agent",
        "tool",
        "mcp_server",
        "mcp_tool",
        "plugin",
        "skill",
        "extension",
    }


def test_mcp_dispatch_contract_cases():
    vectors = json.loads((CONTRACT_DIR / "matching.json").read_text())
    for case in vectors["mcp_dispatch_cases"]:
        catalog = case["catalog"]
        resolution = resolve_mcp_tool_identity(
            case["raw_name"],
            catalog["servers"],
            catalog.get("tools"),
            authoritative_resource=case.get("authoritative_resource"),
        )
        if case["failure"] is None:
            expected = case["result"]
            assert resolution.status == "resolved", case["name"]
            assert resolution.resource == GovernanceResource(
                kind=expected["kind"],
                id=expected["id"],
                parent_id=expected.get("parent_id"),
            ), case["name"]
        else:
            assert resolution.status == case["failure"], case["name"]
            assert resolution.resource is None, case["name"]


@pytest.mark.parametrize(
    "value",
    [
        {"kind": "connector", "id": "x"},
        {"kind": "tool", "id": ""},
        {"kind": "tool", "id": "   "},
        {"kind": "tool", "id": "x", "parent_id": "server"},
        {"kind": "mcp_tool", "id": "x"},
        {"kind": "mcp_tool", "id": "x", "parent_id": ""},
        {"kind": "tool", "id": "x", "extra": True},
    ],
)
def test_malformed_concrete_resources_are_rejected(value):
    with pytest.raises(ValueError):
        GovernanceResource.from_wire(value)


@pytest.mark.parametrize(
    "host_tool_name",
    ["mcp__arxiv_mcp_server__search", "MCP:arxiv_mcp_server:search"],
)
def test_bare_server_catalog_derives_hyphen_host_alias(host_tool_name):
    resolution = resolve_mcp_tool_identity(host_tool_name, ["arxiv-mcp-server"])
    assert resolution.status == "resolved"
    assert resolution.resource == GovernanceResource(
        kind="mcp_tool",
        id="search",
        parent_id="arxiv-mcp-server",
    )


def test_derived_host_alias_collides_with_exact_server_id():
    resolution = resolve_mcp_tool_identity(
        "mcp__arxiv_mcp_server__search",
        ["arxiv-mcp-server", "arxiv_mcp_server"],
    )
    assert resolution.status == "ambiguous"
    assert resolution.resource is None
    colon = resolve_mcp_tool_identity(
        "MCP:arxiv_mcp_server:search",
        ["arxiv_mcp_server", "arxiv-mcp-server"],
    )
    assert colon.status == "ambiguous"
    assert colon.resource is None


def test_explicit_alias_remains_equal_to_derived_host_alias():
    servers = [{"id": "arxiv-mcp-server", "aliases": ["custom_host"]}]
    explicit = resolve_mcp_tool_identity("mcp__custom_host__search", servers)
    derived = resolve_mcp_tool_identity("MCP:arxiv_mcp_server:search", servers)
    assert explicit.status == "resolved"
    assert explicit.resource == GovernanceResource(
        kind="mcp_tool",
        id="search",
        parent_id="arxiv-mcp-server",
    )
    assert derived.status == "resolved"
    assert derived.resource == GovernanceResource(
        kind="mcp_tool",
        id="search",
        parent_id="arxiv-mcp-server",
    )


def test_separator_server_ids_derive_host_alias_without_reversing_underscores():
    derived = resolve_mcp_tool_identity(
        "mcp__my__server__search",
        ["my--server"],
    )
    assert derived.status == "resolved"
    assert derived.resource == GovernanceResource(
        kind="mcp_tool",
        id="search",
        parent_id="my--server",
    )
    reverse = resolve_mcp_tool_identity(
        "mcp__arxiv-mcp-server__search",
        ["arxiv_mcp_server"],
    )
    assert reverse.status == "unresolved"
    assert reverse.resource is None
    unrelated = resolve_mcp_tool_identity(
        "MCP:other:search",
        ["arxiv-mcp-server"],
    )
    assert unrelated.status == "unresolved"
    assert unrelated.resource is None


@pytest.mark.parametrize(
    "host_tool_name",
    [
        "mcp__silmaril_firewall__list",
        "MCP:silmaril_firewall:list",
    ],
)
def test_resolver_supports_both_host_forms_and_unique_normalized_alias(
    host_tool_name,
):
    alias = resolve_mcp_tool_identity(
        host_tool_name,
        [{"id": "silmaril-firewall", "aliases": ["silmaril_firewall"]}],
    )
    assert alias.status == "resolved"
    assert alias.resource == GovernanceResource(
        kind="mcp_tool", id="list", parent_id="silmaril-firewall"
    )


@pytest.mark.parametrize(
    "host_tool_name",
    [
        "mcp__silmaril_firewall__list",
        "MCP:silmaril_firewall:list",
    ],
)
def test_resolver_reports_exact_and_alias_servers_as_ambiguous(host_tool_name):
    resolution = resolve_mcp_tool_identity(
        host_tool_name,
        [
            {"id": "silmaril-firewall", "aliases": ["silmaril_firewall"]},
            "silmaril_firewall",
        ],
    )
    assert resolution.status == "ambiguous"
    assert resolution.resource is None



@pytest.mark.parametrize(
    "host_tool_name",
    ["mcp__a_b_c__list", "MCP:a_b_c:list"],
)
def test_resolver_reports_normalized_alias_collisions(host_tool_name):
    resolution = resolve_mcp_tool_identity(
        host_tool_name,
        [
            {"id": "a-b_c", "aliases": ["a_b_c"]},
            {"id": "a_b-c", "aliases": ["a_b_c"]},
        ],
    )
    assert resolution.status == "ambiguous"
    assert resolution.resource is None


@pytest.mark.parametrize(
    "host_tool_name",
    [
        "read_file",
        "mcp____list",
        "mcp__server__",
        "mcp__   __list",
        "MCP::list",
        "MCP:server:",
        "MCP:server:   ",
    ],
)
def test_resolver_never_falls_back_to_native_or_resolves_empty_ids(host_tool_name):
    resolution = resolve_mcp_tool_identity(
        host_tool_name, ["server", "   "]
    )
    assert resolution.status == "unresolved"
    assert resolution.resource is None


def test_resolver_reports_unknown_configured_server_without_fallback():
    resolution = resolve_mcp_tool_identity(
        "MCP:missing:list", ["configured"]
    )
    assert resolution.status == "unresolved"
    assert resolution.resource is None
    assert resolve_mcp_tool_identity(
        "mcp__missing__list", ["configured"]
    ).status == "unresolved"


@pytest.mark.parametrize(
    ("host_tool_name", "server_id"),
    [
        ("mcp__prod__west__search", "prod__west"),
        ("MCP:prod:west:search", "prod:west"),
    ],
)
def test_resolver_keeps_separator_ids_authoritative(host_tool_name, server_id):
    resolved = resolve_mcp_tool_identity(host_tool_name, [server_id])
    assert resolved.status == "resolved"
    assert resolved.resource == GovernanceResource(
        kind="mcp_tool",
        id="search",
        parent_id=server_id,
    )


@pytest.mark.parametrize(
    ("host_tool_name", "server_id", "tool_id"),
    [
        ("mcp__a__b__c", "a", "b__c"),
        ("MCP:a:b:c", "a", "b:c"),
        ("mcp__prod__west__edge__search", "prod__west", "edge__search"),
    ],
)
def test_unique_server_prefix_keeps_nested_tool_id(host_tool_name, server_id, tool_id):
    resolution = resolve_mcp_tool_identity(host_tool_name, [server_id])
    assert resolution.status == "resolved"
    assert resolution.resource == GovernanceResource(
        kind="mcp_tool",
        id=tool_id,
        parent_id=server_id,
    )


@pytest.mark.parametrize(
    ("host_tool_name", "server_ids"),
    [
        ("mcp__a__b__c", ["a", "a__b"]),
        ("MCP:a:b:c", ["a", "a:b"]),
        ("mcp__prod__west__search", ["prod", "prod__west"]),
    ],
)
def test_overlapping_server_prefixes_are_ambiguous(host_tool_name, server_ids):
    for configured_ids in (server_ids, list(reversed(server_ids))):
        resolution = resolve_mcp_tool_identity(host_tool_name, configured_ids)
        assert resolution.status == "ambiguous"
        assert resolution.resource is None


def test_resolver_keeps_shorter_exact_prefix_when_longer_id_does_not_match():
    resolution = resolve_mcp_tool_identity(
        "mcp__prod__search",
        ["prod__west", "prod"],
    )
    assert resolution.status == "resolved"
    assert resolution.resource == GovernanceResource(
        kind="mcp_tool",
        id="search",
        parent_id="prod",
    )


def test_exact_and_alias_prefixes_are_ambiguous():
    collision = resolve_mcp_tool_identity(
        "mcp__prod_west__search",
        ["prod_west", {"id": "prod-west", "aliases": ["prod_west"]}],
    )
    assert collision.status == "ambiguous"
    assert collision.resource is None
    separator_alias = resolve_mcp_tool_identity(
        "mcp__prod__west__search",
        ["prod__west", {"id": "prod-_west", "aliases": ["prod__west"]}],
    )
    assert separator_alias.status == "ambiguous"
    assert separator_alias.resource is None

    collision = resolve_mcp_tool_identity(
        "mcp__a_b__c__list",
        [
            {"id": "a-b__c", "aliases": ["a_b__c"]},
            {"id": "a-b-_c", "aliases": ["a_b__c"]},
        ],
    )
    assert collision.status == "ambiguous"
    assert collision.resource is None
    colon_collision = resolve_mcp_tool_identity(
        "MCP:a_b:c_d:list",
        [
            {"id": "a-b:c_d", "aliases": ["a_b:c_d"]},
            {"id": "a-b:c-d", "aliases": ["a_b:c_d"]},
        ],
    )
    assert colon_collision.status == "ambiguous"
    assert colon_collision.resource is None


@pytest.mark.parametrize(
    "host_tool_name",
    [
        "search",
        "mcp_prod_west_search",
        "mcp__prod__west__search\n",
        "mcp__prod\n__west__search",
        "mcp__prod__west__search\rsearch",
        "MCP:prod:west:search\n",
        "MCP:prod\n:west:search",
        "mcp__" + ("a_" * 5000) + "search",
    ],
)
def test_resolver_rejects_malformed_newline_and_long_names(host_tool_name):
    resolution = resolve_mcp_tool_identity(
        host_tool_name,
        ["prod__west", "prod:west", "prod\nwest", "a"],
    )
    assert resolution.status == "unresolved"
    assert resolution.resource is None


def test_tool_catalog_matches_complete_spellings_and_dedupes():
    tool = {"id": "b__c", "parent_id": "a"}
    resolved = resolve_mcp_tool_identity(
        "mcp__a__b__c",
        ["a"],
        [tool, tool],
    )
    assert resolved.status == "resolved"
    assert resolved.resource == GovernanceResource(
        kind="mcp_tool",
        id="b__c",
        parent_id="a",
    )
    shorter = resolve_mcp_tool_identity(
        "mcp__a__b__c",
        ["a", "a__b"],
        [{"id": "b", "parent_id": "a"}],
    )
    assert shorter.status == "unresolved"
    assert shorter.resource is None
    missing = resolve_mcp_tool_identity("mcp__a__b", ["a"], [])
    assert missing.status == "unresolved"


def test_repeated_server_rows_keep_every_alias():
    servers = [
        {"id": "git", "aliases": ["origin"]},
        {"id": "git", "aliases": ["docs"]},
    ]
    tools = [
        {"id": "search", "parent_id": "git"},
        {"id": "search__nested", "parent_id": "git"},
        {"id": "search:nested", "parent_id": "git"},
    ]
    cases = [
        ("mcp__origin__search", "search"),
        ("mcp__docs__search", "search"),
        ("MCP:origin:search", "search"),
        ("MCP:docs:search", "search"),
        ("mcp__origin__search__nested", "search__nested"),
        ("mcp__docs__search__nested", "search__nested"),
        ("MCP:origin:search:nested", "search:nested"),
        ("MCP:docs:search:nested", "search:nested"),
    ]
    for configured in (servers, list(reversed(servers))):
        for host_tool_name, tool_id in cases:
            catalog = resolve_mcp_tool_identity(host_tool_name, configured, tools)
            assert catalog.status == "resolved"
            assert catalog.resource == GovernanceResource(
                kind="mcp_tool",
                id=tool_id,
                parent_id="git",
            )
            server_only = resolve_mcp_tool_identity(host_tool_name, configured)
            assert server_only.status == "resolved"
            assert server_only.resource == catalog.resource


def test_shared_alias_across_distinct_servers_stays_ambiguous():
    servers = [
        {"id": "git", "aliases": ["shared"]},
        {"id": "docs", "aliases": ["shared"]},
    ]
    tools = [
        {"id": "search", "parent_id": "git"},
        {"id": "search", "parent_id": "docs"},
        {"id": "search__nested", "parent_id": "git"},
        {"id": "search__nested", "parent_id": "docs"},
        {"id": "search:nested", "parent_id": "git"},
        {"id": "search:nested", "parent_id": "docs"},
    ]
    for host_tool_name in (
        "mcp__shared__search",
        "MCP:shared:search",
        "mcp__shared__search__nested",
        "MCP:shared:search:nested",
    ):
        catalog = resolve_mcp_tool_identity(host_tool_name, servers, tools)
        assert catalog.status == "ambiguous"
        assert catalog.resource is None
        server_only = resolve_mcp_tool_identity(host_tool_name, servers)
        assert server_only.status == "ambiguous"
        assert server_only.resource is None


def test_tool_catalog_orphan_parent_is_unresolved_unless_authoritative():
    stale_tool = {
        "id": "search",
        "parent_id": "removed-server",
        "resource": {
            "kind": "mcp_tool",
            "id": "search",
            "parent_id": "removed-server",
        },
    }
    servers = ["other-server"]
    trusted = GovernanceResource(
        kind="mcp_tool",
        id="search",
        parent_id="removed-server",
    )
    for host_tool_name in (
        "mcp__removed-server__search",
        "MCP:removed-server:search",
    ):
        orphan = resolve_mcp_tool_identity(host_tool_name, servers, [stale_tool])
        assert orphan.status == "unresolved"
        assert orphan.resource is None
        override = resolve_mcp_tool_identity(
            host_tool_name,
            servers,
            [stale_tool],
            authoritative_resource=trusted,
        )
        assert override.status == "resolved"
        assert override.resource == trusted
    server_only = resolve_mcp_tool_identity("mcp__other-server__search", servers)
    assert server_only.status == "resolved"
    assert server_only.resource == GovernanceResource(
        kind="mcp_tool",
        id="search",
        parent_id="other-server",
    )
    colon_only = resolve_mcp_tool_identity("MCP:other-server:search", servers)
    assert colon_only.status == "resolved"
    assert colon_only.resource == server_only.resource


def test_tool_catalog_exact_and_alias_spellings_are_ambiguous():
    resolution = resolve_mcp_tool_identity(
        "MCP:prod_west:search",
        [{"id": "prod-west", "aliases": ["prod_west"]}, "prod_west"],
        [
            {"id": "search", "parent_id": "prod-west"},
            {"id": "search", "parent_id": "prod_west"},
        ],
    )
    assert resolution.status == "ambiguous"
    assert resolution.resource is None


def test_resolver_matches_long_configured_separator_id():
    server_id = "prod__" + ("west__" * 50) + "region"
    resolution = resolve_mcp_tool_identity(
        f"mcp__{server_id}__search",
        [server_id],
    )
    assert resolution.status == "resolved"
    assert resolution.resource == GovernanceResource(
        kind="mcp_tool",
        id="search",
        parent_id=server_id,
    )


def test_sync_resource_wire_response_callback_and_benign_governance_block(monkeypatch):
    payloads = []
    events = []
    firewall = Firewall(
        api_key="sk",
        api_url=TEST_API_URL,
        on_classify=events.append,
    )

    def post(payload):
        payloads.append(payload)
        return {
            "prediction": "BENIGN",
            "score": 0.0,
            "threshold": 0.5,
            "mode": "block",
            "governance": governance_block(),
        }

    monkeypatch.setattr(firewall, "_post_json", post)
    with pytest.raises(FirewallBlockedException) as exc:
        firewall.classify(
            "{}",
            tool_name="mcp__github__create_issue",
            resource=GovernanceResource(
                kind="mcp_tool", id="create_issue", parent_id="github"
            ),
            identity_revision="catalog-9",
            metadata={
                "resource": {"kind": "tool", "id": "spoofed"},
                "identity_revision": "spoofed",
            },
            request_id="request-1",
        )

    assert payloads[0]["tool_name"] == "mcp__github__create_issue"
    assert payloads[0]["resource"] == {
        "kind": "mcp_tool",
        "id": "create_issue",
        "parent_id": "github",
    }
    assert payloads[0]["identity_revision"] == "catalog-9"
    assert payloads[0]["metadata"]["resource"]["id"] == "spoofed"
    assert exc.value.result is not None
    assert exc.value.result.governance == GovernanceDecision(
        action="block",
        rule_id="deny-github",
        policy_version="policy-7",
        resource=GovernanceResource(
            kind="mcp_tool", id="create_issue", parent_id="github"
        ),
        identity_revision="catalog-9",
        reason="identity_unresolved",
    )
    assert events[0].blocked is True


@pytest.mark.parametrize("mode", ["shadow", "warn"])
def test_benign_governance_block_preserves_non_enforcing_mode(monkeypatch, mode):
    firewall = Firewall(api_key="sk", api_url=TEST_API_URL)
    monkeypatch.setattr(
        firewall,
        "_post_json",
        lambda payload: {
            "prediction": "BENIGN",
            "score": 0.0,
            "threshold": 0.5,
            "mode": "block",
            "governance": governance_block(),
        },
    )
    result = firewall.classify("{}", mode=mode)
    assert result.mode == mode
    assert result.governance is not None
    assert result.governance.action == "block"


def test_batch_resources_align_and_governance_blocks(monkeypatch):
    firewall = Firewall(api_key="sk", api_url=TEST_API_URL)
    resource = GovernanceResource(kind="skill", id="review")
    with pytest.raises(ValueError, match="resources length 1"):
        firewall.classify_batch(["a", "b"], resources=[resource])

    payloads = []

    def post(payload):
        payloads.append(payload)
        return {
            "predictions": [
                {
                    "prediction": "BENIGN",
                    "score": 0.0,
                    "threshold": 0.5,
                    "mode": "block",
                    "governance": governance_block(),
                },
                {
                    "prediction": "BENIGN",
                    "score": 0.0,
                    "threshold": 0.5,
                    "mode": "block",
                },
            ]
        }

    monkeypatch.setattr(firewall, "_post_json", post)
    with pytest.raises(BatchFirewallBlockedException) as exc:
        firewall.classify_batch(
            ["a", "b"],
            tool_names=["raw-a", "raw-b"],
            resources=[resource, None],
            identity_revision="catalog-9",
        )
    assert payloads[0]["tool_names"] == ["raw-a", "raw-b"]
    assert payloads[0]["resources"] == [{"kind": "skill", "id": "review"}, None]
    assert payloads[0]["identity_revision"] == "catalog-9"
    assert [item.index for item in exc.value.blocked] == [0]


@pytest.mark.asyncio
async def test_async_resource_wire_and_benign_governance_block():
    payloads = []

    async def handle(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return httpx.Response(
            200,
            request=request,
            json={
                "prediction": "BENIGN",
                "score": 0.0,
                "threshold": 0.5,
                "mode": "block",
                "governance": governance_block(),
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        firewall = AsyncFirewall(
            api_key="sk", api_url=TEST_API_URL, http_client=client
        )
        with pytest.raises(FirewallBlockedException):
            await firewall.classify(
                "{}",
                tool_name="raw-name",
                resource=GovernanceResource(kind="extension", id="publisher.extension"),
                identity_revision="catalog-9",
            )
        await firewall.aclose()

    assert payloads[0]["tool_name"] == "raw-name"
    assert payloads[0]["resource"] == {
        "kind": "extension",
        "id": "publisher.extension",
    }
    assert payloads[0]["identity_revision"] == "catalog-9"


def test_langchain_enforces_benign_governance_block(monkeypatch):
    pytest.importorskip("langchain_core.callbacks")
    firewall = Firewall(api_key="sk", api_url=TEST_API_URL)
    handler = firewall.as_langchain_handler()
    monkeypatch.setattr(
        firewall,
        "_classify_raw",
        lambda *args, **kwargs: BlockResult(
            prediction="BENIGN",
            score=0.0,
            threshold=0.5,
            mode="block",
            governance=GovernanceDecision(
                action="block",
                policy_version="policy-7",
            ),
        ),
    )
    with pytest.raises(FirewallBlockedException):
        handler.on_chat_model_start(
            serialized={},
            messages=[[{"role": "user", "content": "hello"}]],
            run_id=uuid4(),
        )
