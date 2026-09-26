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
    [
        "mcp__silmaril_firewall__list",
        "MCP:silmaril_firewall:list",
    ],
)
def test_resolver_supports_both_host_forms_and_unique_normalized_alias(
    host_tool_name,
):
    alias = resolve_mcp_tool_identity(host_tool_name, ["silmaril-firewall"])
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
def test_resolver_prefers_exact_configured_server(host_tool_name):
    exact = resolve_mcp_tool_identity(
        host_tool_name, ["silmaril-firewall", "silmaril_firewall"]
    )
    assert exact.status == "resolved"
    assert exact.resource == GovernanceResource(
        kind="mcp_tool", id="list", parent_id="silmaril_firewall"
    )



@pytest.mark.parametrize(
    "host_tool_name",
    ["mcp__a_b_c__list", "MCP:a_b_c:list"],
)
def test_resolver_reports_normalized_alias_collisions(host_tool_name):
    resolution = resolve_mcp_tool_identity(
        host_tool_name, ["a-b_c", "a_b-c"]
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
