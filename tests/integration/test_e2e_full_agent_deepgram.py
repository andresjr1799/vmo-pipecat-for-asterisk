"""
Integration tests — full-agent pipeline with Deepgram Voice Agent (Phase 5).

Verifies (§9.4.2):
- context.prompt overrides params.instructions in the agent service
- context.greeting overrides params.greeting
- context.tools are registered on the agent
- Pipeline structure: input → agent → output (no SileroVAD)
- Full call lifecycle: build → start → audio → shutdown → vmo.call.ended
"""

from __future__ import annotations

import asyncio
import struct
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from vmo_pipecat.audio.audiosocket_server import (
    AudioSocketServer, TYPE_UUID, TYPE_AUDIO, TYPE_TERMINATE,
)
from vmo_pipecat.call.controller import CallController
from vmo_pipecat.call.identity import CallIdentity
from vmo_pipecat.call.registry import CallRegistry
from vmo_pipecat.call.router import CallRouter
from vmo_pipecat.config.models import (
    AudioProfileCfg,
    ContextCfg,
    DeepgramVoiceAgentProviderCfg,
    FullAgentPipelineCfg,
    TransferCfg,
    OverridesCfg,
)
from vmo_pipecat.events.bus import LoggingEventBus
from vmo_pipecat.pipelines.factory import PipelineFactory
from vmo_pipecat.tenancy.resolver import SessionConfig
from vmo_pipecat.transport.asterisk_transport import AsteriskAudioSocketTransport


# ── Helpers ────────────────────────────────────────────────────────────────────

def _tlv(msg_type: int, payload: bytes) -> bytes:
    return bytes([msg_type]) + struct.pack(">H", len(payload)) + payload


def _dg_session(
    prompt: str = "You are a support agent.",
    greeting: str = "Hello, how can I help?",
    tools: list | None = None,
    provider_params: dict | None = None,
) -> SessionConfig:
    """Build a full-agent session config using Deepgram Voice Agent."""
    provider_params = provider_params or {
        "model": "nova-2",
        "tts_model": "aura-2-thalia-en",
    }
    pipeline = FullAgentPipelineCfg(kind="full_agent", provider="dg_va")
    return SessionConfig(
        context=ContextCfg(
            prompt=prompt,
            greeting=greeting,
            audio_profile="telephony_8k",
            tools=tools or [],
        ),
        pipeline=pipeline,
        providers={
            "dg_va": DeepgramVoiceAgentProviderCfg(
                kind="deepgram_voice_agent",
                mode="full_agent",
                api_key="dg-test-key",
                params=provider_params,
            ),
        },
        audio_profile=AudioProfileCfg(in_rate=8000, out_rate=8000),
        transfer=TransferCfg(enabled=True, context="from-vmo-transfer", default_target="9000"),
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


# ── Unit tests: builder behaviour (§9.4.2) ────────────────────────────────────

def test_deepgram_builder_sets_instructions_from_context_prompt():
    """context.prompt must override params.instructions (§9.4.2)."""
    from vmo_pipecat.providers.deepgram_voice_agent import build_service
    from vmo_pipecat.config.models import DeepgramVoiceAgentProviderCfg, AudioProfileCfg

    resolved = DeepgramVoiceAgentProviderCfg(
        kind="deepgram_voice_agent", mode="full_agent", api_key="key",
        params={"instructions": "default instructions"},
    )
    service = build_service(resolved, AudioProfileCfg(), context_prompt="override prompt")
    # Service stores the resolved kwargs; verify override happened
    assert service._kw.get("instructions") == "override prompt"


def test_deepgram_builder_sets_greeting_from_context():
    from vmo_pipecat.providers.deepgram_voice_agent import build_service
    from vmo_pipecat.config.models import DeepgramVoiceAgentProviderCfg, AudioProfileCfg

    resolved = DeepgramVoiceAgentProviderCfg(
        kind="deepgram_voice_agent", mode="full_agent", api_key="key",
        params={"greeting": "default hello"},
    )
    service = build_service(resolved, AudioProfileCfg(), context_greeting="context greeting")
    assert service._kw.get("greeting") == "context greeting"


def test_deepgram_builder_preserves_params_when_no_context_override():
    from vmo_pipecat.providers.deepgram_voice_agent import build_service
    from vmo_pipecat.config.models import DeepgramVoiceAgentProviderCfg, AudioProfileCfg

    resolved = DeepgramVoiceAgentProviderCfg(
        kind="deepgram_voice_agent", mode="full_agent", api_key="key",
        params={"model": "nova-2", "tts_model": "aura-2-thalia-en"},
    )
    service = build_service(resolved, AudioProfileCfg())
    assert service._kw.get("model") == "nova-2"
    assert service._kw.get("tts_model") == "aura-2-thalia-en"


def test_full_agent_pipeline_structure():
    """Pipeline must be [input, agent, output] — no SileroVAD."""
    from vmo_pipecat.pipelines.full_agent import build_full_agent_pipeline
    from vmo_pipecat.actions.asterisk_actions import AsteriskActions
    from vmo_pipecat.events.bus import LoggingEventBus

    session = _dg_session()
    transport = MagicMock()
    transport.input.return_value = "input_proc"
    transport.output.return_value = "output_proc"

    pool = MagicMock()
    identity = CallIdentity(
        vmo_call_id="c1", asterisk_channel_id="ch-1", call_id_sbc="s1",
        tenant_id="acme", tenant_name="Acme", node_id="ast-1", did="1000",
    )
    actions = AsteriskActions(pool, identity, "from-vmo-transfer", LoggingEventBus())
    runner, task = build_full_agent_pipeline(session, transport, actions)

    # Pipeline should have exactly 3 processors: input, agent, output
    assert len(task.pipeline.processors) == 3
    assert task.pipeline.processors[0] == "input_proc"
    assert task.pipeline.processors[2] == "output_proc"


def test_full_agent_tools_registered_on_agent():
    """context.tools must be registered on the agent service."""
    from vmo_pipecat.pipelines.full_agent import build_full_agent_pipeline
    from vmo_pipecat.actions.asterisk_actions import AsteriskActions
    from vmo_pipecat.events.bus import LoggingEventBus
    from vmo_pipecat.config.models import DeepgramVoiceAgentProviderCfg

    session = _dg_session(tools=["transfer_call", "hangup_call"])
    transport = MagicMock()
    transport.input.return_value = MagicMock()
    transport.output.return_value = MagicMock()

    registered = []

    class TrackingAgent:
        def register_function(self, name, cb): registered.append(name)

    from vmo_pipecat.providers import registry as _reg
    orig = _reg.PROVIDER_BUILDERS.get("deepgram_voice_agent")
    _reg.PROVIDER_BUILDERS["deepgram_voice_agent"] = lambda r, p, **kw: TrackingAgent()
    try:
        pool = MagicMock()
        identity = CallIdentity(
            vmo_call_id="c2", asterisk_channel_id="ch-2", call_id_sbc="s2",
            tenant_id="acme", tenant_name="Acme", node_id="ast-1", did="1000",
        )
        actions = AsteriskActions(pool, identity, "from-vmo-transfer", LoggingEventBus())
        build_full_agent_pipeline(session, transport, actions)
    finally:
        if orig:
            _reg.PROVIDER_BUILDERS["deepgram_voice_agent"] = orig
        else:
            _reg.PROVIDER_BUILDERS.pop("deepgram_voice_agent", None)

    assert set(registered) == {"transfer_call", "hangup_call"}


# ── Integration: full call lifecycle ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_full_agent_call_lifecycle(stack, capsys):
    """Full call: build DG Voice Agent pipeline, connect, then hangup."""
    server, router, registry, event_bus = stack
    pool, ari_client = _mock_pool()
    session = _dg_session(prompt="Agent prompt", greeting="Hola!")

    audio_uuid = str(uuid.uuid4())
    identity = CallIdentity(
        vmo_call_id=str(uuid.uuid4()),
        asterisk_channel_id="SIP/dg-001",
        call_id_sbc="sbc-dg-1",
        tenant_id="acme",
        tenant_name="Acme",
        node_id="ast-1",
        did="1001",
    )
    transport = AsteriskAudioSocketTransport(server, session.audio_profile)
    controller = CallController(
        identity=identity,
        session_config=session,
        bridge_id="bridge-dg",
        pool=pool,
        audiosocket=server,
        transport=transport,
        router=router,
        registry=registry,
        event_bus=event_bus,
    )

    await router.register_pending(audio_uuid, controller)
    await registry.add(controller)
    task = asyncio.create_task(controller.start(), name="ctrl-dg")

    # Connect and perform UUID handshake
    reader, writer = await asyncio.open_connection("127.0.0.1", server.port)
    try:
        writer.write(_tlv(TYPE_UUID, uuid.UUID(audio_uuid).bytes))
        await writer.drain()
        await asyncio.sleep(0.05)

        assert registry.active_count == 1

        # Initiate hangup (simulates caller hanging up)
        await controller.shutdown(reason="stasis_end")
        await asyncio.sleep(0.05)

        assert registry.active_count == 0

        # Verify vmo.call.ended was emitted
        import json
        out = capsys.readouterr().out
        events = [json.loads(ln[6:]) for ln in out.splitlines() if ln.startswith("EVENT ")]
        ended = [e for e in events if e["subject"] == "vmo.call.ended"]
        assert ended, "Expected vmo.call.ended"
        assert ended[-1]["outcome"] == "stasis_end"

    finally:
        writer.close()
        if not task.done():
            task.cancel()
        await asyncio.sleep(0.05)


@pytest.mark.asyncio
async def test_full_agent_pipeline_emit_vmo_call_pipeline_started(stack, capsys):
    """vmo.call.pipeline.started must carry pipeline_kind=full_agent."""
    server, router, registry, event_bus = stack
    pool, _ = _mock_pool()
    session = _dg_session()

    audio_uuid = str(uuid.uuid4())
    identity = CallIdentity(
        vmo_call_id=str(uuid.uuid4()),
        asterisk_channel_id="SIP/dg-002",
        call_id_sbc="sbc-dg-2",
        tenant_id="acme",
        tenant_name="Acme",
        node_id="ast-1",
        did="1001",
    )
    transport = AsteriskAudioSocketTransport(server, session.audio_profile)
    controller = CallController(
        identity=identity,
        session_config=session,
        bridge_id="bridge-dg2",
        pool=pool,
        audiosocket=server,
        transport=transport,
        router=router,
        registry=registry,
        event_bus=event_bus,
    )

    await registry.add(controller)
    task = asyncio.create_task(controller.start(), name="ctrl-dg2")
    await asyncio.sleep(0.05)

    await controller.shutdown(reason="stasis_end")
    await asyncio.sleep(0.05)

    import json
    out = capsys.readouterr().out
    events = [json.loads(ln[6:]) for ln in out.splitlines() if ln.startswith("EVENT ")]
    started = [e for e in events if e["subject"] == "vmo.call.pipeline.started"]
    assert started, "Expected vmo.call.pipeline.started"
    assert started[0]["pipeline_kind"] == "full_agent"

    if not task.done():
        task.cancel()
    await asyncio.sleep(0.05)
