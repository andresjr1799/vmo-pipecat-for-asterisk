"""
Integration tests — Asterisk events → PipeCat frame injection (Phase 8).

Covers §4.2:
  ChannelDtmfReceived (ARI)     → DTMFFrame queued into pipeline task
  AudioSocket TYPE_DTMF         → same path via CallRouter
  ChannelTalkingStarted (ARI)   → UserStartedSpeakingFrame (vad=asterisk_talk_detect only)
  ChannelTalkingFinished (ARI)  → UserStoppedSpeakingFrame (vad=asterisk_talk_detect only)
  Caller hangup during TTS      → pipeline cancels < 200 ms (defensive hangup)
"""

from __future__ import annotations

import asyncio
import struct
import time
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vmo_pipecat.audio.audiosocket_server import (
    AudioSocketServer, TYPE_UUID, TYPE_DTMF,
)
from vmo_pipecat.call.controller import CallController
from vmo_pipecat.call.identity import CallIdentity
from vmo_pipecat.call.registry import CallRegistry
from vmo_pipecat.call.router import CallRouter
from vmo_pipecat.config.models import (
    AudioProfileCfg,
    ContextCfg,
    DeepgramProviderCfg,
    ElevenLabsProviderCfg,
    ModularPipelineCfg,
    OpenAIProviderCfg,
    TransferCfg,
    OverridesCfg,
)
from vmo_pipecat.events.bus import LoggingEventBus
from vmo_pipecat.events.from_asterisk import DTMFFrame, UserStartedSpeakingFrame, UserStoppedSpeakingFrame
from vmo_pipecat.tenancy.resolver import SessionConfig
from vmo_pipecat.transport.asterisk_transport import AsteriskAudioSocketTransport


# ── Helpers ────────────────────────────────────────────────────────────────────

def _tlv(msg_type: int, payload: bytes) -> bytes:
    return bytes([msg_type]) + struct.pack(">H", len(payload)) + payload


def _identity(channel_id: str = "SIP/trunk-test") -> CallIdentity:
    return CallIdentity(
        vmo_call_id=str(uuid.uuid4()),
        asterisk_channel_id=channel_id,
        call_id_sbc="sbc-1",
        tenant_id="acme",
        tenant_name="Acme",
        node_id="ast-1",
        did="1000",
    )


def _modular_session(vad: str = "silero") -> SessionConfig:
    pipeline = ModularPipelineCfg(
        kind="modular", stt="stt_p", llm="llm_p", tts="tts_p", vad=vad
    )
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


def _mock_pool():
    pool = MagicMock()
    client = MagicMock()
    client.hangup_channel = AsyncMock()
    pool.client_for.return_value = client
    return pool, client


@pytest.fixture
async def stack():
    router = CallRouter()
    registry = CallRegistry()
    event_bus = LoggingEventBus()
    server = AudioSocketServer(
        host="127.0.0.1", port=0,
        on_uuid=lambda conn_id, uid: router.bind_uuid(uid, conn_id),
        on_audio=router.dispatch_audio,
        on_disconnect=router.dispatch_disconnect,
        on_dtmf=router.dispatch_dtmf,
    )
    await server.start()
    yield server, router, registry, event_bus
    await server.stop()


def _make_controller(server, router, registry, event_bus, session=None, channel_id=None):
    pool, _ = _mock_pool()
    session = session or _modular_session()
    identity = _identity(channel_id or "SIP/trunk-test")
    transport = AsteriskAudioSocketTransport(server, session.audio_profile)
    return CallController(
        identity=identity,
        session_config=session,
        bridge_id="bridge-1",
        pool=pool,
        audiosocket=server,
        transport=transport,
        router=router,
        registry=registry,
        event_bus=event_bus,
    )


# ═════════════════════════════════════════════════════════════════════════════
# DTMFFrame injection
# ═════════════════════════════════════════════════════════════════════════════

class TestDtmfFrameInjection:

    def test_dtmf_frame_dataclass(self):
        frame = DTMFFrame(digit="5", channel_id="ch-1")
        assert frame.digit == "5"
        assert frame.channel_id == "ch-1"

    @pytest.mark.asyncio
    async def test_on_dtmf_queues_frame_to_pipeline_task(self, stack):
        """controller.on_dtmf() must queue DTMFFrame into the pipeline task."""
        server, router, registry, event_bus = stack
        session = _modular_session()
        ctrl = _make_controller(server, router, registry, event_bus, session)

        queued_frames = []
        ctrl._echo_mode = False

        # Mock the task
        mock_task = MagicMock()
        mock_task.queue_frame = AsyncMock(side_effect=lambda f: queued_frames.append(f))
        ctrl._task = mock_task

        await ctrl.on_dtmf("7")

        assert len(queued_frames) == 1
        frame = queued_frames[0]
        assert isinstance(frame, DTMFFrame)
        assert frame.digit == "7"
        assert frame.channel_id == "SIP/trunk-test"

    @pytest.mark.asyncio
    async def test_on_dtmf_echo_mode_does_not_queue(self, stack):
        """In echo mode, DTMF is logged only — no frame injection."""
        server, router, registry, event_bus = stack
        ctrl = _make_controller(server, router, registry, event_bus)
        ctrl._echo_mode = True

        mock_task = MagicMock()
        mock_task.queue_frame = AsyncMock()
        ctrl._task = mock_task

        await ctrl.on_dtmf("5")

        mock_task.queue_frame.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_on_dtmf_no_task_does_not_raise(self, stack):
        """When no pipeline task exists yet, on_dtmf must be a silent no-op."""
        server, router, registry, event_bus = stack
        ctrl = _make_controller(server, router, registry, event_bus)
        ctrl._echo_mode = False
        ctrl._task = None

        await ctrl.on_dtmf("3")   # must not raise

    @pytest.mark.asyncio
    async def test_ari_dtmf_event_reaches_controller_via_lifecycle(self, stack):
        """Simulate ARI ChannelDtmfReceived: lifecycle → registry → controller."""
        from vmo_pipecat.call.lifecycle import CallLifecycle
        from vmo_pipecat.ari.events import CHANNEL_DTMF_RECEIVED

        server, router, registry, event_bus = stack
        pool, _ = _mock_pool()
        session = _modular_session()
        channel_id = "SIP/lifecycle-dtmf"
        ctrl = _make_controller(server, router, registry, event_bus, session, channel_id)
        ctrl._echo_mode = False

        queued = []
        mock_task = MagicMock()
        mock_task.queue_frame = AsyncMock(side_effect=lambda f: queued.append(f))
        ctrl._task = mock_task

        await registry.add(ctrl)
        router.register_by_channel(channel_id, ctrl)

        # Build minimal lifecycle and call the handler directly
        resolver = MagicMock()
        lifecycle = CallLifecycle(
            pool=pool, audiosocket=server, router=router, registry=registry,
            resolver=resolver, event_bus=event_bus,
        )

        ari_event = {
            "type": CHANNEL_DTMF_RECEIVED,
            "channel": {"id": channel_id, "name": "SIP/trunk", "channelvars": {}},
            "digit": "9",
            "_vmo_node_id": "ast-1",
        }
        await lifecycle._on_dtmf_received(ari_event)

        assert len(queued) == 1
        assert isinstance(queued[0], DTMFFrame)
        assert queued[0].digit == "9"

    @pytest.mark.asyncio
    async def test_audiosocket_dtmf_reaches_controller_via_router(self, stack):
        """AudioSocket TYPE_DTMF → CallRouter.dispatch_dtmf → controller.on_dtmf."""
        server, router, registry, event_bus = stack
        session = _modular_session()
        ctrl = _make_controller(server, router, registry, event_bus, session)
        ctrl._echo_mode = False

        queued = []
        mock_task = MagicMock()
        mock_task.queue_frame = AsyncMock(side_effect=lambda f: queued.append(f))
        ctrl._task = mock_task

        audio_uuid = str(uuid.uuid4())
        await router.register_pending(audio_uuid, ctrl)
        await registry.add(ctrl)

        reader, writer = await asyncio.open_connection("127.0.0.1", server.port)
        try:
            writer.write(_tlv(TYPE_UUID, uuid.UUID(audio_uuid).bytes))
            await writer.drain()
            await asyncio.sleep(0.05)

            # Send DTMF '4' via AudioSocket TLV
            writer.write(_tlv(TYPE_DTMF, b"4"))
            await writer.drain()
            await asyncio.sleep(0.05)

            assert any(isinstance(f, DTMFFrame) and f.digit == "4" for f in queued)
        finally:
            writer.close()
            await asyncio.sleep(0.05)


# ═════════════════════════════════════════════════════════════════════════════
# ChannelTalking → Speaking frames (vad=asterisk_talk_detect only)
# ═════════════════════════════════════════════════════════════════════════════

class TestChannelTalkingFrames:

    @pytest.mark.asyncio
    async def test_talking_started_queues_user_started_frame_when_vad_is_talk_detect(self, stack):
        server, router, registry, event_bus = stack
        session = _modular_session(vad="asterisk_talk_detect")
        ctrl = _make_controller(server, router, registry, event_bus, session)
        ctrl._echo_mode = False

        queued = []
        mock_task = MagicMock()
        mock_task.queue_frame = AsyncMock(side_effect=lambda f: queued.append(f))
        ctrl._task = mock_task

        await ctrl.on_talking_started()

        assert len(queued) == 1
        assert isinstance(queued[0], UserStartedSpeakingFrame)

    @pytest.mark.asyncio
    async def test_talking_finished_queues_user_stopped_frame_when_vad_is_talk_detect(self, stack):
        server, router, registry, event_bus = stack
        session = _modular_session(vad="asterisk_talk_detect")
        ctrl = _make_controller(server, router, registry, event_bus, session)
        ctrl._echo_mode = False

        queued = []
        mock_task = MagicMock()
        mock_task.queue_frame = AsyncMock(side_effect=lambda f: queued.append(f))
        ctrl._task = mock_task

        await ctrl.on_talking_finished()

        assert len(queued) == 1
        assert isinstance(queued[0], UserStoppedSpeakingFrame)

    @pytest.mark.asyncio
    async def test_talking_ignored_when_vad_is_silero(self, stack):
        """Default (Silero VAD): ChannelTalking events are silently ignored."""
        server, router, registry, event_bus = stack
        session = _modular_session(vad="silero")
        ctrl = _make_controller(server, router, registry, event_bus, session)
        ctrl._echo_mode = False

        mock_task = MagicMock()
        mock_task.queue_frame = AsyncMock()
        ctrl._task = mock_task

        await ctrl.on_talking_started()
        await ctrl.on_talking_finished()

        mock_task.queue_frame.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_talking_ignored_when_vad_none(self, stack):
        server, router, registry, event_bus = stack
        session = _modular_session(vad="none")
        ctrl = _make_controller(server, router, registry, event_bus, session)
        ctrl._echo_mode = False

        mock_task = MagicMock()
        mock_task.queue_frame = AsyncMock()
        ctrl._task = mock_task

        await ctrl.on_talking_started()
        mock_task.queue_frame.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_lifecycle_dispatches_talking_started_to_controller(self, stack):
        """CallLifecycle._on_channel_talking_started → registry → controller."""
        from vmo_pipecat.call.lifecycle import CallLifecycle

        server, router, registry, event_bus = stack
        pool, _ = _mock_pool()
        session = _modular_session(vad="asterisk_talk_detect")
        channel_id = "SIP/talking-test"
        ctrl = _make_controller(server, router, registry, event_bus, session, channel_id)
        ctrl._echo_mode = False

        queued = []
        mock_task = MagicMock()
        mock_task.queue_frame = AsyncMock(side_effect=lambda f: queued.append(f))
        ctrl._task = mock_task

        await registry.add(ctrl)
        router.register_by_channel(channel_id, ctrl)

        lifecycle = CallLifecycle(
            pool=pool, audiosocket=server, router=router, registry=registry,
            resolver=MagicMock(), event_bus=event_bus,
        )
        ari_event = {
            "channel": {"id": channel_id, "name": "SIP/trunk", "channelvars": {}},
            "_vmo_node_id": "ast-1",
        }
        await lifecycle._on_channel_talking_started(ari_event)

        assert any(isinstance(f, UserStartedSpeakingFrame) for f in queued)


# ═════════════════════════════════════════════════════════════════════════════
# Defensive hangup: pipeline cancels < 200 ms
# ═════════════════════════════════════════════════════════════════════════════

class TestDefensiveHangup:

    @pytest.mark.asyncio
    async def test_stasis_end_cancels_pipeline_within_200ms(self, stack):
        """StasisEnd while pipeline is running must cancel it in < 200 ms."""
        server, router, registry, event_bus = stack
        pool, _ = _mock_pool()
        session = _modular_session()
        ctrl = _make_controller(server, router, registry, event_bus, session)

        await registry.add(ctrl)
        ctrl_task = asyncio.create_task(ctrl.start(), name="ctrl-hangup-test")
        await asyncio.sleep(0.05)   # let pipeline stub start blocking

        t0 = time.monotonic()
        await ctrl.shutdown(reason="stasis_end")
        elapsed_ms = (time.monotonic() - t0) * 1000

        assert elapsed_ms < 200, f"Shutdown took {elapsed_ms:.1f} ms (> 200 ms)"
        assert registry.active_count == 0

        if not ctrl_task.done():
            ctrl_task.cancel()
        await asyncio.sleep(0.05)

    @pytest.mark.asyncio
    async def test_shutdown_cancels_pipeline_task(self, stack):
        """shutdown() must call task.cancel() on the PipeCat task."""
        server, router, registry, event_bus = stack
        session = _modular_session()
        ctrl = _make_controller(server, router, registry, event_bus, session)
        ctrl._echo_mode = False

        cancel_called = []
        mock_task = MagicMock()
        mock_task.cancel = AsyncMock(side_effect=lambda: cancel_called.append(True))
        ctrl._task = mock_task

        await registry.add(ctrl)
        await ctrl.shutdown(reason="stasis_end")

        assert cancel_called, "task.cancel() was not called during shutdown"

    @pytest.mark.asyncio
    async def test_shutdown_idempotent_does_not_double_cancel(self, stack):
        """Calling shutdown() twice must cancel the pipeline only once."""
        server, router, registry, event_bus = stack
        session = _modular_session()
        ctrl = _make_controller(server, router, registry, event_bus, session)
        ctrl._echo_mode = False

        cancel_count = []
        mock_task = MagicMock()
        mock_task.cancel = AsyncMock(side_effect=lambda: cancel_count.append(1))
        ctrl._task = mock_task

        await registry.add(ctrl)
        await ctrl.shutdown(reason="stasis_end")
        await ctrl.shutdown(reason="stasis_end")   # second call: idempotent

        assert len(cancel_count) == 1

    @pytest.mark.asyncio
    async def test_audiosocket_disconnect_triggers_shutdown(self, stack):
        """AudioSocket on_disconnect must initiate shutdown."""
        server, router, registry, event_bus = stack
        session = _modular_session()
        ctrl = _make_controller(server, router, registry, event_bus, session)

        audio_uuid = str(uuid.uuid4())
        await router.register_pending(audio_uuid, ctrl)
        await registry.add(ctrl)
        ctrl_task = asyncio.create_task(ctrl.start(), name="ctrl-disconnect-test")

        reader, writer = await asyncio.open_connection("127.0.0.1", server.port)
        try:
            writer.write(_tlv(TYPE_UUID, uuid.UUID(audio_uuid).bytes))
            await writer.drain()
            await asyncio.sleep(0.05)

            # Close connection abruptly — triggers on_disconnect
            writer.close()
            await asyncio.sleep(0.15)

            assert registry.active_count == 0

        finally:
            if not ctrl_task.done():
                ctrl_task.cancel()
            await asyncio.sleep(0.05)
