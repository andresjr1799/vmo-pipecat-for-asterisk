"""
Unit tests for PipelineFactory, provider registry, and AsteriskActions.

These tests mock PipeCat services so they run without pipecat-ai installed.
They verify: correct provider selection, action registration, tool schema
injection, and factory dispatch by pipeline.kind.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from vmo_pipecat.actions.tool_schemas import schemas_for_tools, ALL_SCHEMAS
from vmo_pipecat.config.models import (
    AudioProfileCfg,
    ContextCfg,
    ModularPipelineCfg,
    FullAgentPipelineCfg,
    DeepgramProviderCfg,
    ElevenLabsProviderCfg,
    OpenAIProviderCfg,
    DeepgramVoiceAgentProviderCfg,
    TransferCfg,
    OverridesCfg,
)
from vmo_pipecat.tenancy.resolver import SessionConfig


# ── Helpers ────────────────────────────────────────────────────────────────────

def _audio() -> AudioProfileCfg:
    return AudioProfileCfg(in_rate=8000, out_rate=8000)


def _modular_session(tools=None) -> SessionConfig:
    pipeline = ModularPipelineCfg(kind="modular", stt="stt_p", llm="llm_p", tts="tts_p")
    return SessionConfig(
        context=ContextCfg(
            prompt="Eres un agente.",
            greeting="¡Hola!",
            audio_profile="telephony_8k",
            tools=tools or [],
        ),
        pipeline=pipeline,
        providers={
            "stt_p": DeepgramProviderCfg(kind="deepgram", mode="stt", api_key="dk"),
            "llm_p": OpenAIProviderCfg(kind="openai", mode="llm", api_key="ok"),
            "tts_p": ElevenLabsProviderCfg(kind="elevenlabs", mode="tts", api_key="ek"),
        },
        audio_profile=_audio(),
        transfer=TransferCfg(enabled=True, context="from-vmo-transfer", default_target="9000"),
        overrides=OverridesCfg(),
        config_version=1,
    )


def _full_agent_session() -> SessionConfig:
    pipeline = FullAgentPipelineCfg(kind="full_agent", provider="dg_va")
    return SessionConfig(
        context=ContextCfg(
            prompt="You are a voice agent.",
            greeting="Hello!",
            audio_profile="telephony_8k",
            tools=["transfer_call"],
        ),
        pipeline=pipeline,
        providers={
            "dg_va": DeepgramVoiceAgentProviderCfg(
                kind="deepgram_voice_agent", mode="full_agent", api_key="dg"
            ),
        },
        audio_profile=_audio(),
        transfer=TransferCfg(),
        overrides=OverridesCfg(),
        config_version=1,
    )


def _mock_transport():
    t = MagicMock()
    t.input.return_value = MagicMock()
    t.output.return_value = MagicMock()
    t.push_audio = AsyncMock()
    return t


def _mock_actions():
    a = MagicMock()
    a.transfer_call = AsyncMock()
    a.hangup_call = AsyncMock()
    a.play_audio_file = AsyncMock()
    a.send_dtmf = AsyncMock()
    return a


# ── Tool schemas ───────────────────────────────────────────────────────────────

def test_all_schemas_defined():
    assert "transfer_call" in ALL_SCHEMAS
    assert "hangup_call" in ALL_SCHEMAS
    assert "play_audio_file" in ALL_SCHEMAS
    assert "send_dtmf" in ALL_SCHEMAS


def test_schemas_for_tools_filters_correctly():
    result = schemas_for_tools(["transfer_call", "hangup_call"])
    assert len(result) == 2
    names = {s["function"]["name"] for s in result}
    assert names == {"transfer_call", "hangup_call"}


def test_schemas_for_tools_ignores_unknown():
    result = schemas_for_tools(["transfer_call", "nonexistent_tool"])
    assert len(result) == 1


def test_schemas_openai_format():
    for name, schema in ALL_SCHEMAS.items():
        assert schema["type"] == "function"
        fn = schema["function"]
        assert "name" in fn
        assert "description" in fn
        assert "parameters" in fn
        assert fn["parameters"]["type"] == "object"
        assert "properties" in fn["parameters"]


# ── Provider registry ──────────────────────────────────────────────────────────

def test_build_provider_deepgram_stt():
    from vmo_pipecat.providers.registry import build_provider
    cfg = DeepgramProviderCfg(kind="deepgram", mode="stt", api_key="key")
    service = build_provider(cfg, _audio())
    assert service is not None


def test_build_provider_deepgram_tts():
    from vmo_pipecat.providers.registry import build_provider
    cfg = DeepgramProviderCfg(kind="deepgram", mode="tts", api_key="key")
    service = build_provider(cfg, _audio())
    assert service is not None


def test_build_provider_openai():
    from vmo_pipecat.providers.registry import build_provider
    cfg = OpenAIProviderCfg(kind="openai", mode="llm", api_key="key",
                            params={"model": "gpt-4o-mini"})
    service = build_provider(cfg, _audio())
    assert service is not None


def test_build_provider_elevenlabs():
    from vmo_pipecat.providers.registry import build_provider
    cfg = ElevenLabsProviderCfg(kind="elevenlabs", mode="tts", api_key="key",
                                params={"voice_id": "v1"})
    service = build_provider(cfg, _audio())
    assert service is not None


def test_build_provider_unknown_kind_raises():
    from vmo_pipecat.providers.registry import build_provider

    class FakeCfg:
        kind = "nonexistent_kind"

    with pytest.raises(ValueError, match="Unknown provider kind"):
        build_provider(FakeCfg(), _audio())


def test_register_custom_provider():
    from vmo_pipecat.providers.registry import register, build_provider, PROVIDER_BUILDERS
    original = PROVIDER_BUILDERS.pop("test_custom", None)
    try:
        register("test_custom", lambda resolved, profile: "custom_service")

        class FakeCfg:
            kind = "test_custom"

        result = build_provider(FakeCfg(), _audio())
        assert result == "custom_service"
    finally:
        PROVIDER_BUILDERS.pop("test_custom", None)
        if original:
            PROVIDER_BUILDERS["test_custom"] = original


# ── PipelineFactory dispatch ───────────────────────────────────────────────────

def test_factory_dispatch_modular():
    from vmo_pipecat.pipelines.factory import PipelineFactory
    session = _modular_session()
    transport = _mock_transport()
    actions = _mock_actions()

    runner, task = PipelineFactory.build(session, transport, actions)
    assert runner is not None
    assert task is not None


def test_factory_dispatch_full_agent():
    from vmo_pipecat.pipelines.factory import PipelineFactory
    session = _full_agent_session()
    transport = _mock_transport()
    actions = _mock_actions()

    runner, task = PipelineFactory.build(session, transport, actions)
    assert runner is not None
    assert task is not None


def test_factory_unknown_kind_raises():
    from vmo_pipecat.pipelines.factory import PipelineFactory
    session = _modular_session()
    session = session.__class__(
        context=session.context,
        pipeline=MagicMock(kind="nonexistent"),
        providers=session.providers,
        audio_profile=session.audio_profile,
        transfer=session.transfer,
        overrides=session.overrides,
        config_version=1,
    )
    with pytest.raises(ValueError, match="Unknown pipeline kind"):
        PipelineFactory.build(session, _mock_transport(), _mock_actions())


# ── Modular pipeline: tool registration ───────────────────────────────────────

def test_modular_registers_listed_tools_on_llm():
    """Verify that only context.tools are registered on the LLM service."""
    from vmo_pipecat.pipelines.modular import build_modular_pipeline
    from vmo_pipecat.providers.registry import register, PROVIDER_BUILDERS

    registered_tools = []

    class MockLLM:
        def register_function(self, name, cb):
            registered_tools.append(name)

    # Temporarily replace openai builder to return our mock
    orig = PROVIDER_BUILDERS.get("openai")
    PROVIDER_BUILDERS["openai"] = lambda r, p: MockLLM()
    try:
        session = _modular_session(tools=["transfer_call", "hangup_call"])
        transport = _mock_transport()
        actions = _mock_actions()
        build_modular_pipeline(session, transport, actions)
    finally:
        if orig:
            PROVIDER_BUILDERS["openai"] = orig
        else:
            PROVIDER_BUILDERS.pop("openai", None)

    assert set(registered_tools) == {"transfer_call", "hangup_call"}


def test_modular_no_tools_registers_nothing():
    from vmo_pipecat.pipelines.modular import build_modular_pipeline
    from vmo_pipecat.providers.registry import PROVIDER_BUILDERS

    registered_tools = []

    class MockLLM:
        def register_function(self, name, cb):
            registered_tools.append(name)

    orig = PROVIDER_BUILDERS.get("openai")
    PROVIDER_BUILDERS["openai"] = lambda r, p: MockLLM()
    try:
        session = _modular_session(tools=[])
        build_modular_pipeline(session, _mock_transport(), _mock_actions())
    finally:
        if orig:
            PROVIDER_BUILDERS["openai"] = orig
        else:
            PROVIDER_BUILDERS.pop("openai", None)

    assert registered_tools == []


# ── AsteriskActions ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_transfer_call_invokes_continue_in_dialplan():
    from vmo_pipecat.actions.asterisk_actions import AsteriskActions
    from vmo_pipecat.call.identity import CallIdentity
    from vmo_pipecat.events.bus import LoggingEventBus

    pool = MagicMock()
    ari_client = MagicMock()
    ari_client.continue_in_dialplan = AsyncMock(return_value=True)
    pool.client_for.return_value = ari_client

    identity = CallIdentity(
        vmo_call_id="c1", asterisk_channel_id="ch-1", call_id_sbc="sbc-1",
        tenant_id="acme", tenant_name="Acme", node_id="ast-1", did="1000",
    )

    result_holder = []
    async def _cb(r): result_holder.append(r)

    actions = AsteriskActions(pool, identity, "from-vmo-transfer", LoggingEventBus())
    await actions.transfer_call("transfer_call", "tc1", {"target": "9000"}, None, None, _cb)

    ari_client.continue_in_dialplan.assert_awaited_once_with(
        "ch-1", context="from-vmo-transfer", extension="9000", priority=1
    )
    assert result_holder[0]["status"] == "ok"


@pytest.mark.asyncio
async def test_transfer_call_idempotent():
    from vmo_pipecat.actions.asterisk_actions import AsteriskActions
    from vmo_pipecat.call.identity import CallIdentity
    from vmo_pipecat.events.bus import LoggingEventBus

    pool = MagicMock()
    ari_client = MagicMock()
    ari_client.continue_in_dialplan = AsyncMock(return_value=True)
    pool.client_for.return_value = ari_client

    identity = CallIdentity(
        vmo_call_id="c2", asterisk_channel_id="ch-2", call_id_sbc="sbc-2",
        tenant_id="acme", tenant_name="Acme", node_id="ast-1", did="1000",
    )
    called = []

    async def cb(r): called.append(r)

    actions = AsteriskActions(pool, identity, "from-vmo-transfer", LoggingEventBus())
    await actions.transfer_call("transfer_call", "tc1", {"target": "9000"}, None, None, cb)
    await actions.transfer_call("transfer_call", "tc2", {"target": "9000"}, None, None, cb)

    # continue_in_dialplan only called once (idempotency guard)
    assert ari_client.continue_in_dialplan.await_count == 1
    assert called[1] == {"status": "already_transferred"}
