"""Unit tests for EventBus and LoggingEventBus."""

import json
import pytest
from unittest.mock import patch
from io import StringIO

from vmo_pipecat.events.bus import EventBus, LoggingEventBus
from vmo_pipecat.call.identity import CallIdentity


def _identity() -> CallIdentity:
    return CallIdentity(
        vmo_call_id="call-1",
        asterisk_channel_id="ch-1",
        call_id_sbc="sbc-1",
        tenant_id="acme",
        tenant_name="Acme S.A.",
        node_id="ast-1",
        did="1000",
    )


@pytest.mark.asyncio
async def test_logging_bus_implements_protocol():
    bus = LoggingEventBus()
    assert isinstance(bus, EventBus)


@pytest.mark.asyncio
async def test_logging_bus_emits_json(capsys):
    bus = LoggingEventBus()
    identity = _identity()
    await bus.emit("vmo.call.started", identity, did="1000", extra="value")

    captured = capsys.readouterr().out
    assert captured.startswith("EVENT ")

    data = json.loads(captured[len("EVENT "):])
    assert data["subject"] == "vmo.call.started"
    assert data["vmo_call_id"] == "call-1"
    assert data["tenant_id"] == "acme"
    assert data["did"] == "1000"
    assert data["extra"] == "value"
    assert "ts" in data


@pytest.mark.asyncio
async def test_logging_bus_without_identity(capsys):
    bus = LoggingEventBus()
    await bus.emit("vmo.system.config.reloaded", config_version=3)

    captured = capsys.readouterr().out
    data = json.loads(captured[len("EVENT "):])
    assert data["subject"] == "vmo.system.config.reloaded"
    assert data["config_version"] == 3
    assert "vmo_call_id" not in data


@pytest.mark.asyncio
async def test_logging_bus_close_is_noop():
    bus = LoggingEventBus()
    await bus.close()  # should not raise


@pytest.mark.asyncio
async def test_logging_bus_all_identity_keys_present(capsys):
    bus = LoggingEventBus()
    identity = _identity()
    await bus.emit("vmo.call.ended", identity, outcome="completed", duration_ms=5000)

    captured = capsys.readouterr().out
    data = json.loads(captured[len("EVENT "):])

    for key in ("vmo_call_id", "asterisk_channel_id", "call_id_sbc",
                "tenant_id", "tenant_name", "node_id", "did"):
        assert key in data, f"Missing key: {key}"
