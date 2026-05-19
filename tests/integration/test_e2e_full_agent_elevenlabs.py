"""
Integration tests — full-agent pipeline with ElevenLabs Conversational (Phase 5).

Verifies (§9.4.2):
- context.prompt → system_prompt override in ElevenLabs service
- context.greeting → first_message override
- context.tools registered on agent
- Pipeline structure: input → agent → output
- Full call lifecycle with clean shutdown
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
    ElevenLabsConvProviderCfg,
    FullAgentPipelineCfg,
    TransferCfg,
    OverridesCfg,
)
from vmo_pipecat.events.bus import LoggingEventBus
from vmo_pipecat.tenancy.resolver import SessionConfig
from vmo_pipecat.transport.asterisk_transport import AsteriskAudioSocketTransport


# ── Helpers ────────────────────────────────────────────────────────────────────

def _tlv(msg_type: int, payload: bytes) -> bytes:
    return bytes([msg_type]) + struct.pack(">H", len(payload)) + payload


def _el_session(
    prompt: str = "Eres un agente de ventas.",
    greeting: str = "¡Hola! ¿En qué te ayudo?",
    tools: list | None = None,
) -> SessionConfig:
    pipeline = FullAgentPipelineCfg(kind="full_agent", provider="el_conv")
    return SessionConfig(
        context=ContextCfg(
            prompt=prompt,
            greeting=greeting,
            audio_profile="telephony_8k",
            tools=tools or [],
        ),
        pipeline=pipeline,
        providers={
            "el_conv": ElevenLabsConvProviderCfg(
                kind="elevenlabs_conv",
                mode="full_agent",
                api_key="el-test-key",
                params={"agent_id": "agent-globex-001"},
            ),
        },
        audio_profile=AudioProfileCfg(in_rate=8000, out_rate=8000),
        transfer=TransferCfg(enabled=True, context="from-vmo-transfer", default_target="8000"),
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

def test_elevenlabs_builder_sets_system_prompt_from_context():
    """context.prompt must become system_prompt (§9.4.2 ElevenLabs)."""
    from vmo_pipecat.providers.elevenlabs_conv import build_service

    resolved = ElevenLabsConvProviderCfg(
        kind="elevenlabs_conv", mode="full_agent", api_key="key",
        params={"agent_id": "agent-001"},
    )
    service = build_service(
        resolved, AudioProfileCfg(),
        context_prompt="Context override prompt",
    )
    assert service._kw.get("system_prompt") == "Context override prompt"


def test_elevenlabs_builder_sets_first_message_from_context():
    """context.greeting must become first_message (§9.4.2 ElevenLabs)."""
    from vmo_pipecat.providers.elevenlabs_conv import build_service

    resolved = ElevenLabsConvProviderCfg(
        kind="elevenlabs_conv", mode="full_agent", api_key="key",
        params={"agent_id": "agent-001"},
    )
    service = build_service(
        resolved, AudioProfileCfg(),
        context_greeting="Context override greeting",
    )
    assert service._kw.get("first_message") == "Context override greeting"


def test_elevenlabs_builder_passes_agent_id():
    from vmo_pipecat.providers.elevenlabs_conv import build_service

    resolved = ElevenLabsConvProviderCfg(
        kind="elevenlabs_conv", mode="full_agent", api_key="key",
        params={"agent_id": "my-agent-id"},
    )
    service = build_service(resolved, AudioProfileCfg())
    # agent_id is passed as positional kwarg to the service
    assert service._kw.get("agent_id") == "my-agent-id"


def test_elevenlabs_builder_no_context_override_when_empty():
    """Empty context fields must NOT override provider params."""
    from vmo_pipecat.providers.elevenlabs_conv import build_service

    resolved = ElevenLabsConvProviderCfg(
        kind="elevenlabs_conv", mode="full_agent", api_key="key",
        params={"agent_id": "agent-001"},
    )
    # No context_prompt or context_greeting → no system_prompt/first_message in kwargs
    service = build_service(resolved, AudioProfileCfg())
    assert "system_prompt" not in service._kw
    assert "first_message" not in service._kw


def test_elevenlabs_full_agent_pipeline_structure():
    from vmo_pipecat.pipelines.full_agent import build_full_agent_pipeline
    from vmo_pipecat.actions.asterisk_actions import AsteriskActions

    session = _el_session()
    transport = MagicMock()
    transport.input.return_value = "in"
    transport.output.return_value = "out"

    pool = MagicMock()
    identity = CallIdentity(
        vmo_call_id="c3", asterisk_channel_id="ch-3", call_id_sbc="s3",
        tenant_id="globex", tenant_name="Globex", node_id="ast-1", did="default",
    )
    actions = AsteriskActions(pool, identity, "from-vmo-transfer", LoggingEventBus())
    _, task = build_full_agent_pipeline(session, transport, actions)

    # [input, agent, output] — no VAD
    assert len(task.pipeline.processors) == 3
    assert task.pipeline.processors[0] == "in"
    assert task.pipeline.processors[2] == "out"


def test_elevenlabs_tools_registered():
    from vmo_pipecat.pipelines.full_agent import build_full_agent_pipeline
    from vmo_pipecat.actions.asterisk_actions import AsteriskActions
    from vmo_pipecat.providers import registry as _reg

    session = _el_session(tools=["transfer_call"])
    transport = MagicMock()
    transport.input.return_value = MagicMock()
    transport.output.return_value = MagicMock()

    registered = []

    class TrackingAgent:
        def register_function(self, name, cb):
            registered.append(name)

    orig = _reg.PROVIDER_BUILDERS.get("elevenlabs_conv")
    _reg.PROVIDER_BUILDERS["elevenlabs_conv"] = lambda r, p, **kw: TrackingAgent()
    try:
        pool = MagicMock()
        identity = CallIdentity(
            vmo_call_id="c4", asterisk_channel_id="ch-4", call_id_sbc="s4",
            tenant_id="globex", tenant_name="Globex", node_id="ast-1", did="default",
        )
        actions = AsteriskActions(pool, identity, "from-vmo-transfer", LoggingEventBus())
        build_full_agent_pipeline(session, transport, actions)
    finally:
        if orig:
            _reg.PROVIDER_BUILDERS["elevenlabs_conv"] = orig
        else:
            _reg.PROVIDER_BUILDERS.pop("elevenlabs_conv", None)

    assert "transfer_call" in registered


# ── Integration: full call lifecycle ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_elevenlabs_full_call_lifecycle(stack, capsys):
    server, router, registry, event_bus = stack
    pool, _ = _mock_pool()
    session = _el_session(prompt="Eres Globex assistant.", greeting="¡Hola Globex!")

    audio_uuid = str(uuid.uuid4())
    identity = CallIdentity(
        vmo_call_id=str(uuid.uuid4()),
        asterisk_channel_id="SIP/el-001",
        call_id_sbc="sbc-el-1",
        tenant_id="globex",
        tenant_name="Globex Corp",
        node_id="ast-1",
        did="default",
    )
    transport = AsteriskAudioSocketTransport(server, session.audio_profile)
    controller = CallController(
        identity=identity,
        session_config=session,
        bridge_id="bridge-el",
        pool=pool,
        audiosocket=server,
        transport=transport,
        router=router,
        registry=registry,
        event_bus=event_bus,
    )

    await router.register_pending(audio_uuid, controller)
    await registry.add(controller)
    task = asyncio.create_task(controller.start(), name="ctrl-el")

    reader, writer = await asyncio.open_connection("127.0.0.1", server.port)
    try:
        writer.write(_tlv(TYPE_UUID, uuid.UUID(audio_uuid).bytes))
        await writer.drain()
        await asyncio.sleep(0.05)

        assert registry.active_count == 1

        await controller.shutdown(reason="stasis_end")
        await asyncio.sleep(0.05)

        assert registry.active_count == 0

        import json
        out = capsys.readouterr().out
        events = [json.loads(ln[6:]) for ln in out.splitlines() if ln.startswith("EVENT ")]
        ended = [e for e in events if e["subject"] == "vmo.call.ended"]
        assert ended
        assert ended[-1]["tenant_id"] == "globex"
        assert ended[-1]["outcome"] == "stasis_end"

    finally:
        writer.close()
        if not task.done():
            task.cancel()
        await asyncio.sleep(0.05)
