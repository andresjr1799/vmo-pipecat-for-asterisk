"""
Integration test: Phase 3 echo — full audio path without real Asterisk.

Simulates the flow:
  1. CallLifecycle._on_caller_channel() sets up controller + registers pending
  2. AudioSocket TLV client sends UUID handshake (CallRouter.bind_uuid)
  3. Client sends TYPE_AUDIO → controller.on_audio() → echo back
  4. Client reads echoed audio from socket
  5. Hangup (StasisEnd) → controller.shutdown() → vmo.call.ended emitted
"""

from __future__ import annotations

import asyncio
import struct
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vmo_pipecat.audio.audiosocket_server import AudioSocketServer, TYPE_UUID, TYPE_AUDIO, TYPE_TERMINATE
from vmo_pipecat.call.controller import CallController
from vmo_pipecat.call.identity import CallIdentity
from vmo_pipecat.call.registry import CallRegistry
from vmo_pipecat.call.router import CallRouter
from vmo_pipecat.config.models import (
    AudioProfileCfg, DeepgramProviderCfg, ElevenLabsProviderCfg, OpenAIProviderCfg,
)
from vmo_pipecat.events.bus import LoggingEventBus
from vmo_pipecat.transport.asterisk_transport import AsteriskAudioSocketTransport


# ── Helpers ────────────────────────────────────────────────────────────────────

def _tlv(msg_type: int, payload: bytes) -> bytes:
    return bytes([msg_type]) + struct.pack(">H", len(payload)) + payload


def _mock_ari_pool():
    pool = MagicMock()
    client = MagicMock()
    client.hangup_channel = AsyncMock()
    client.answer_channel = AsyncMock()
    client.create_bridge = AsyncMock(return_value="bridge-test")
    client.add_channel_to_bridge = AsyncMock()
    client.send_command = AsyncMock(return_value={"id": "new-channel"})
    pool.client_for = MagicMock(return_value=client)
    pool.is_any_connected = False
    return pool, client


def _mock_session_config():
    from vmo_pipecat.config.models import (
        ContextCfg, ModularPipelineCfg, TransferCfg, OverridesCfg,
    )
    from vmo_pipecat.tenancy.resolver import SessionConfig

    # Provider IDs must match pipeline references so factory can resolve them
    pipeline = ModularPipelineCfg(kind="modular", stt="stt_p", llm="llm_p", tts="tts_p")
    return SessionConfig(
        context=ContextCfg(
            prompt="Hi", greeting="Hello!", audio_profile="telephony_8k", tools=[],
        ),
        pipeline=pipeline,
        providers={
            "stt_p": DeepgramProviderCfg(kind="deepgram", mode="stt", api_key="dk"),
            "llm_p": OpenAIProviderCfg(kind="openai", mode="llm", api_key="ok"),
            "tts_p": ElevenLabsProviderCfg(kind="elevenlabs", mode="tts", api_key="ek"),
        },
        audio_profile=AudioProfileCfg(in_rate=8000, out_rate=8000),
        transfer=TransferCfg(),
        overrides=OverridesCfg(),
        config_version=1,
    )


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
async def stack():
    """Start AudioSocketServer + CallRouter + CallRegistry + EventBus."""
    router = CallRouter()
    registry = CallRegistry()
    event_bus = LoggingEventBus()

    server = AudioSocketServer(
        host="127.0.0.1",
        port=0,
        on_uuid=lambda conn_id, uid: router.bind_uuid(uid, conn_id),
        on_audio=router.dispatch_audio,
        on_disconnect=router.dispatch_disconnect,
        on_dtmf=router.dispatch_dtmf,
    )
    await server.start()
    yield server, router, registry, event_bus
    await server.stop()


# ── Tests ──────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_audio_echoes_back(stack):
    """Bytes sent in → identical bytes received back (echo mode path).

    This test exercises the controller.on_audio() echo path directly,
    bypassing start() so _echo_mode is not overridden by the pipeline builder.
    The UUID handshake is performed over real TCP; audio dispatch goes through
    the real CallRouter + AudioSocketServer chain.
    """
    server, router, registry, event_bus = stack
    pool, ari_client = _mock_ari_pool()
    session_config = _mock_session_config()

    audio_uuid = str(uuid.uuid4())
    identity = CallIdentity(
        vmo_call_id=str(uuid.uuid4()),
        asterisk_channel_id="SIP/trunk-001",
        call_id_sbc="sbc-1",
        tenant_id="acme",
        tenant_name="Acme",
        node_id="ast-1",
        did="1000",
    )

    transport = AsteriskAudioSocketTransport(server, session_config.audio_profile)
    controller = CallController(
        identity=identity,
        session_config=session_config,
        bridge_id="bridge-1",
        pool=pool,
        audiosocket=server,
        transport=transport,
        router=router,
        registry=registry,
        event_bus=event_bus,
    )
    # Echo mode: on_audio() writes directly to the output transport.
    # We do NOT call start() so the pipeline builder cannot override this flag.
    controller._echo_mode = True
    controller._shutdown_event.set()  # mark "started" so shutdown() is stable

    await router.register_pending(audio_uuid, controller)
    await registry.add(controller)

    # TCP client: send UUID handshake
    reader, writer = await asyncio.open_connection("127.0.0.1", server.port)
    try:
        uid_bytes = uuid.UUID(audio_uuid).bytes
        writer.write(_tlv(TYPE_UUID, uid_bytes))
        await writer.drain()
        await asyncio.sleep(0.05)   # let server process handshake

        assert registry.active_count == 1

        # Send audio frames
        payload = bytes(range(160))
        writer.write(_tlv(TYPE_AUDIO, payload))
        await writer.drain()
        await asyncio.sleep(0.05)

        # Read echoed TYPE_AUDIO frame
        header = await asyncio.wait_for(reader.readexactly(3), timeout=1.0)
        msg_type = header[0]
        length = int.from_bytes(header[1:], "big")
        echoed = await asyncio.wait_for(reader.readexactly(length), timeout=1.0)

        assert msg_type == TYPE_AUDIO
        assert echoed == payload
    finally:
        writer.close()
        await asyncio.sleep(0.05)


@pytest.mark.asyncio
async def test_controller_shutdown_on_stasis_end(stack, capsys):
    """Shutdown emits vmo.call.ended with correct outcome."""
    server, router, registry, event_bus = stack
    pool, _ = _mock_ari_pool()
    session_config = _mock_session_config()

    audio_uuid = str(uuid.uuid4())
    identity = CallIdentity(
        vmo_call_id=str(uuid.uuid4()),
        asterisk_channel_id="SIP/trunk-002",
        call_id_sbc="sbc-2",
        tenant_id="acme",
        tenant_name="Acme",
        node_id="ast-1",
        did="1000",
    )
    transport = AsteriskAudioSocketTransport(server, session_config.audio_profile)
    controller = CallController(
        identity=identity,
        session_config=session_config,
        bridge_id="bridge-2",
        pool=pool,
        audiosocket=server,
        transport=transport,
        router=router,
        registry=registry,
        event_bus=event_bus,
    )

    await registry.add(controller)
    asyncio.create_task(controller.start(), name="ctrl-test2")
    await asyncio.sleep(0.01)

    assert registry.active_count == 1
    await controller.shutdown(reason="stasis_end")
    await asyncio.sleep(0.01)

    assert registry.active_count == 0

    # Verify vmo.call.ended was emitted
    out = capsys.readouterr().out
    import json as _json
    events = [_json.loads(line[6:]) for line in out.splitlines() if line.startswith("EVENT ")]
    ended = [e for e in events if e["subject"] == "vmo.call.ended"]
    assert ended, "Expected vmo.call.ended event"
    assert ended[-1]["outcome"] == "stasis_end"
    assert ended[-1]["vmo_call_id"] == identity.vmo_call_id


@pytest.mark.asyncio
async def test_disconnect_triggers_shutdown(stack):
    """AudioSocket disconnect → controller.on_disconnect() → shutdown."""
    server, router, registry, event_bus = stack
    pool, _ = _mock_ari_pool()
    session_config = _mock_session_config()

    audio_uuid = str(uuid.uuid4())
    identity = CallIdentity(
        vmo_call_id=str(uuid.uuid4()),
        asterisk_channel_id="SIP/trunk-003",
        call_id_sbc="sbc-3",
        tenant_id="acme",
        tenant_name="Acme",
        node_id="ast-1",
        did="1000",
    )
    transport = AsteriskAudioSocketTransport(server, session_config.audio_profile)
    controller = CallController(
        identity=identity,
        session_config=session_config,
        bridge_id="bridge-3",
        pool=pool,
        audiosocket=server,
        transport=transport,
        router=router,
        registry=registry,
        event_bus=event_bus,
    )

    await router.register_pending(audio_uuid, controller)
    await registry.add(controller)
    asyncio.create_task(controller.start(), name="ctrl-test3")

    reader, writer = await asyncio.open_connection("127.0.0.1", server.port)
    try:
        writer.write(_tlv(TYPE_UUID, uuid.UUID(audio_uuid).bytes))
        await writer.drain()
        await asyncio.sleep(0.05)
    finally:
        writer.close()
        await asyncio.sleep(0.1)   # give server time to detect disconnect

    assert registry.active_count == 0


@pytest.mark.asyncio
async def test_idempotent_shutdown(stack):
    """Calling shutdown() twice must not crash."""
    server, router, registry, event_bus = stack
    pool, _ = _mock_ari_pool()
    session_config = _mock_session_config()

    identity = CallIdentity(
        vmo_call_id=str(uuid.uuid4()),
        asterisk_channel_id="SIP/trunk-004",
        call_id_sbc="sbc-4",
        tenant_id="acme",
        tenant_name="Acme",
        node_id="ast-1",
        did="1000",
    )
    transport = AsteriskAudioSocketTransport(server, session_config.audio_profile)
    controller = CallController(
        identity=identity,
        session_config=session_config,
        bridge_id="bridge-4",
        pool=pool,
        audiosocket=server,
        transport=transport,
        router=router,
        registry=registry,
        event_bus=event_bus,
    )

    await registry.add(controller)
    asyncio.create_task(controller.start(), name="ctrl-test4")
    await asyncio.sleep(0.01)

    await controller.shutdown(reason="stasis_end")
    await controller.shutdown(reason="stasis_end")   # second call: no-op
    assert registry.active_count == 0
