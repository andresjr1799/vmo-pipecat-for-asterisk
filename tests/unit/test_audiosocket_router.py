"""
Unit tests for CallRouter.

These tests use a real AudioSocketServer bound to 127.0.0.1:0 and a mock TLV
client, plus mock controllers.  No PipeCat dependency.
"""

from __future__ import annotations

import asyncio
import struct
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from vmo_pipecat.call.router import CallRouter
from vmo_pipecat.audio.audiosocket_server import (
    AudioSocketServer,
    TYPE_UUID,
    TYPE_AUDIO,
    TYPE_DTMF,
    TYPE_TERMINATE,
)


# ── TLV helpers ────────────────────────────────────────────────────────────────

def _tlv(msg_type: int, payload: bytes) -> bytes:
    return bytes([msg_type]) + struct.pack(">H", len(payload)) + payload


async def _connect_client(host: str, port: int) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    return await asyncio.open_connection(host, port)


async def _send_uuid(writer: asyncio.StreamWriter, audio_uuid: str) -> None:
    uid_bytes = uuid.UUID(audio_uuid).bytes
    writer.write(_tlv(TYPE_UUID, uid_bytes))
    await writer.drain()


async def _send_audio(writer: asyncio.StreamWriter, audio: bytes) -> None:
    writer.write(_tlv(TYPE_AUDIO, audio))
    await writer.drain()


async def _send_dtmf(writer: asyncio.StreamWriter, digit: str) -> None:
    writer.write(_tlv(TYPE_DTMF, digit.encode("ascii")))
    await writer.drain()


async def _send_terminate(writer: asyncio.StreamWriter) -> None:
    writer.write(_tlv(TYPE_TERMINATE, b""))
    await writer.drain()


# ── Mock controller ────────────────────────────────────────────────────────────

def _mock_controller() -> MagicMock:
    ctrl = MagicMock()
    ctrl.bind_audio_conn = AsyncMock()
    ctrl.on_audio = AsyncMock()
    ctrl.on_dtmf = AsyncMock()
    ctrl.on_disconnect = AsyncMock()
    return ctrl


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def router() -> CallRouter:
    return CallRouter()


@pytest.fixture
async def server_and_router():
    """Starts a real AudioSocketServer wired to a CallRouter."""
    router = CallRouter()

    server = AudioSocketServer(
        host="127.0.0.1",
        port=0,
        on_uuid=lambda conn_id, uid: router.bind_uuid(uid, conn_id),
        on_audio=router.dispatch_audio,
        on_disconnect=router.dispatch_disconnect,
        on_dtmf=router.dispatch_dtmf,
    )
    await server.start()
    yield server, router, server.port
    await server.stop()


# ── CallRouter unit tests (no TCP, no AudioSocketServer) ──────────────────────

@pytest.mark.asyncio
async def test_register_and_bind_uuid(router):
    ctrl = _mock_controller()
    uid = str(uuid.uuid4())
    await router.register_pending(uid, ctrl)
    assert router.pending_count == 1

    ok = await router.bind_uuid(uid, "conn-1")
    assert ok is True
    assert router.pending_count == 0
    assert router.active_count == 1
    ctrl.bind_audio_conn.assert_awaited_once_with("conn-1")


@pytest.mark.asyncio
async def test_bind_unknown_uuid_returns_false(router):
    ok = await router.bind_uuid("no-such-uuid", "conn-x")
    assert ok is False
    assert router.active_count == 0


@pytest.mark.asyncio
async def test_dispatch_audio_reaches_controller(router):
    ctrl = _mock_controller()
    uid = str(uuid.uuid4())
    await router.register_pending(uid, ctrl)
    await router.bind_uuid(uid, "conn-2")

    audio = b"\x00\x01" * 80
    await router.dispatch_audio("conn-2", audio)
    ctrl.on_audio.assert_awaited_once_with(audio)


@pytest.mark.asyncio
async def test_dispatch_audio_unknown_conn_is_silent(router):
    # Should not raise — just silently drops
    await router.dispatch_audio("no-such-conn", b"data")


@pytest.mark.asyncio
async def test_dispatch_dtmf_reaches_controller(router):
    ctrl = _mock_controller()
    uid = str(uuid.uuid4())
    await router.register_pending(uid, ctrl)
    await router.bind_uuid(uid, "conn-3")

    await router.dispatch_dtmf("conn-3", "5")
    ctrl.on_dtmf.assert_awaited_once_with("5")


@pytest.mark.asyncio
async def test_dispatch_disconnect_calls_controller_and_removes(router):
    ctrl = _mock_controller()
    uid = str(uuid.uuid4())
    await router.register_pending(uid, ctrl)
    await router.bind_uuid(uid, "conn-4")
    assert router.active_count == 1

    await router.dispatch_disconnect("conn-4")
    ctrl.on_disconnect.assert_awaited_once()
    assert router.active_count == 0


@pytest.mark.asyncio
async def test_secondary_lookup_vmo_call_id(router):
    ctrl = _mock_controller()
    call_id = str(uuid.uuid4())
    router.register_by_vmo_call_id(call_id, ctrl)
    assert router.get_by_vmo_call_id(call_id) is ctrl


@pytest.mark.asyncio
async def test_secondary_lookup_asterisk_channel(router):
    ctrl = _mock_controller()
    channel_id = "SIP/trunk-000001"
    router.register_by_channel(channel_id, ctrl)
    assert router.get_by_channel(channel_id) is ctrl


@pytest.mark.asyncio
async def test_remove_cleans_secondary_maps(router):
    ctrl = _mock_controller()
    call_id = str(uuid.uuid4())
    channel_id = "SIP/trunk-000002"
    router.register_by_vmo_call_id(call_id, ctrl)
    router.register_by_channel(channel_id, ctrl)

    router.remove(ctrl)
    assert router.get_by_vmo_call_id(call_id) is None
    assert router.get_by_channel(channel_id) is None


@pytest.mark.asyncio
async def test_multiple_concurrent_calls(router):
    ctrls = [_mock_controller() for _ in range(3)]
    uuids = [str(uuid.uuid4()) for _ in range(3)]
    conns = ["conn-a", "conn-b", "conn-c"]

    for uid, ctrl in zip(uuids, ctrls):
        await router.register_pending(uid, ctrl)
    for uid, conn in zip(uuids, conns):
        await router.bind_uuid(uid, conn)

    assert router.active_count == 3

    audio = b"\xff" * 160
    await router.dispatch_audio("conn-b", audio)
    ctrls[0].on_audio.assert_not_awaited()
    ctrls[1].on_audio.assert_awaited_once_with(audio)
    ctrls[2].on_audio.assert_not_awaited()


# ── Integration: real AudioSocketServer + CallRouter via TLV ─────────────────

@pytest.mark.asyncio
async def test_tlv_uuid_handshake_binds_controller(server_and_router):
    server, router, port = server_and_router
    ctrl = _mock_controller()
    uid = str(uuid.uuid4())
    await router.register_pending(uid, ctrl)

    reader, writer = await _connect_client("127.0.0.1", port)
    try:
        await _send_uuid(writer, uid)
        await asyncio.sleep(0.05)   # let server process
        ctrl.bind_audio_conn.assert_awaited_once()
        assert router.active_count == 1
    finally:
        writer.close()
        await asyncio.sleep(0.05)


@pytest.mark.asyncio
async def test_tlv_audio_dispatched_to_controller(server_and_router):
    server, router, port = server_and_router
    ctrl = _mock_controller()
    uid = str(uuid.uuid4())
    await router.register_pending(uid, ctrl)

    reader, writer = await _connect_client("127.0.0.1", port)
    try:
        await _send_uuid(writer, uid)
        await asyncio.sleep(0.05)

        audio = bytes(range(160))
        await _send_audio(writer, audio)
        await asyncio.sleep(0.05)
        ctrl.on_audio.assert_awaited_once_with(audio)
    finally:
        writer.close()
        await asyncio.sleep(0.05)


@pytest.mark.asyncio
async def test_tlv_dtmf_dispatched_to_controller(server_and_router):
    server, router, port = server_and_router
    ctrl = _mock_controller()
    uid = str(uuid.uuid4())
    await router.register_pending(uid, ctrl)

    reader, writer = await _connect_client("127.0.0.1", port)
    try:
        await _send_uuid(writer, uid)
        await asyncio.sleep(0.05)

        await _send_dtmf(writer, "9")
        await asyncio.sleep(0.05)
        ctrl.on_dtmf.assert_awaited_once_with("9")
    finally:
        writer.close()
        await asyncio.sleep(0.05)


@pytest.mark.asyncio
async def test_tlv_terminate_triggers_disconnect(server_and_router):
    server, router, port = server_and_router
    ctrl = _mock_controller()
    uid = str(uuid.uuid4())
    await router.register_pending(uid, ctrl)

    reader, writer = await _connect_client("127.0.0.1", port)
    try:
        await _send_uuid(writer, uid)
        await asyncio.sleep(0.05)
        await _send_terminate(writer)
        await asyncio.sleep(0.05)
        ctrl.on_disconnect.assert_awaited_once()
    finally:
        writer.close()
        await asyncio.sleep(0.05)


@pytest.mark.asyncio
async def test_tlv_unknown_uuid_rejected(server_and_router):
    server, router, port = server_and_router
    reader, writer = await _connect_client("127.0.0.1", port)
    try:
        unknown_uid = str(uuid.uuid4())
        # router has no pending controller for this UUID
        await _send_uuid(writer, unknown_uid)
        await asyncio.sleep(0.05)
        # Connection should receive an error frame and be dropped
        assert router.active_count == 0
    finally:
        writer.close()
        await asyncio.sleep(0.05)


@pytest.mark.asyncio
async def test_send_audio_from_server_to_client(server_and_router):
    """AudioSocketServer.send_audio writes TYPE_AUDIO TLV that the client reads."""
    server, router, port = server_and_router
    ctrl = _mock_controller()
    uid = str(uuid.uuid4())

    # Capture conn_id when controller.bind_audio_conn is called
    bound_conn: list[str] = []
    async def _bind(conn_id: str):
        bound_conn.append(conn_id)
    ctrl.bind_audio_conn = AsyncMock(side_effect=_bind)

    await router.register_pending(uid, ctrl)

    reader, writer = await _connect_client("127.0.0.1", port)
    try:
        await _send_uuid(writer, uid)
        await asyncio.sleep(0.05)
        assert len(bound_conn) == 1

        payload = b"\xAB\xCD" * 80
        await server.send_audio(bound_conn[0], payload)
        await asyncio.sleep(0.05)

        # Read the TYPE_AUDIO TLV frame from the client reader
        header = await asyncio.wait_for(reader.readexactly(3), timeout=1.0)
        msg_type = header[0]
        length = int.from_bytes(header[1:], "big")
        received = await asyncio.wait_for(reader.readexactly(length), timeout=1.0)

        assert msg_type == TYPE_AUDIO
        assert received == payload
    finally:
        writer.close()
        await asyncio.sleep(0.05)
