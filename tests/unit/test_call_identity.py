"""Unit tests for CallIdentity."""

import pytest
from vmo_pipecat.call.identity import CallIdentity


def _make() -> CallIdentity:
    return CallIdentity(
        vmo_call_id="uuid-1",
        asterisk_channel_id="SIP/trunk-001",
        call_id_sbc="sbc-xyz",
        tenant_id="acme",
        tenant_name="Acme S.A.",
        node_id="ast-1",
        did="1000",
    )


def test_frozen():
    identity = _make()
    with pytest.raises((AttributeError, TypeError)):
        identity.vmo_call_id = "other"  # type: ignore[misc]


def test_asdict_contains_all_keys():
    identity = _make()
    d = identity.asdict()
    assert d["vmo_call_id"] == "uuid-1"
    assert d["asterisk_channel_id"] == "SIP/trunk-001"
    assert d["call_id_sbc"] == "sbc-xyz"
    assert d["tenant_id"] == "acme"
    assert d["tenant_name"] == "Acme S.A."
    assert d["node_id"] == "ast-1"
    assert d["did"] == "1000"
    assert d["caller_id"] == ""
    assert len(d) == 8


def test_equality():
    a = _make()
    b = _make()
    assert a == b


def test_hash():
    a = _make()
    b = _make()
    s = {a, b}
    assert len(s) == 1
