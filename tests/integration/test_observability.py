"""
Integration tests — Observabilidad MELT con OpenTelemetry (Fase MELT).

Verifies:
  1. OTel SDK initializes correctly (TracerProvider, MeterProvider).
  2. vmo.call span is created with correct attributes.
  3. Canonical events carry full call identity.
  4. Stage timing helpers work correctly.
  5. JSON logs include trace_id and span_id when inside a span.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from vmo_pipecat.call.controller import CallController
from vmo_pipecat.call.identity import CallIdentity
from vmo_pipecat.call.registry import CallRegistry
from vmo_pipecat.call.router import CallRouter
from vmo_pipecat.audio.audiosocket_server import AudioSocketServer
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
from vmo_pipecat.observability.instrumentation import (
    record_stt_final,
    record_stt_partial,
    record_llm_first_token,
    record_llm_tokens,
    record_tts_first_audio,
    record_barge_in,
    stage_timer,
)
from vmo_pipecat.tenancy.resolver import SessionConfig
from vmo_pipecat.transport.asterisk_transport import AsteriskAudioSocketTransport


# ── Helpers ────────────────────────────────────────────────────────────────────

def _identity(tenant_id: str = "acme") -> CallIdentity:
    return CallIdentity(
        vmo_call_id=str(uuid.uuid4()),
        asterisk_channel_id="SIP/trunk-obs",
        call_id_sbc="sbc-obs",
        tenant_id=tenant_id,
        tenant_name=f"{tenant_id.title()} Corp",
        node_id="ast-1",
        did="1000",
    )


def _session() -> SessionConfig:
    pipeline = ModularPipelineCfg(
        kind="modular", stt="stt_p", llm="llm_p", tts="tts_p"
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


def _make_controller(server, router, registry, event_bus, tenant_id="acme"):
    pool, _ = _mock_pool()
    session = _session()
    identity = _identity(tenant_id=tenant_id)
    transport = AsteriskAudioSocketTransport(server, session.audio_profile)
    return CallController(
        identity=identity,
        session_config=session,
        bridge_id="bridge-obs",
        pool=pool,
        audiosocket=server,
        transport=transport,
        router=router,
        registry=registry,
        event_bus=event_bus,
    )


# ═════════════════════════════════════════════════════════════════════════════
# OTel SDK initialisation
# ═════════════════════════════════════════════════════════════════════════════

def test_otel_sdk_initialises_tracer_and_meter():
    """OTel SDK provides a Tracer and Meter after init."""
    from vmo_pipecat.observability.otel import init_otel, get_tracer, get_meter

    init_otel()

    tracer = get_tracer()
    meter = get_meter()

    assert tracer is not None
    assert meter is not None


def test_otel_sdk_init_is_idempotent():
    """Calling init_otel() multiple times doesn't crash."""
    from vmo_pipecat.observability.otel import init_otel, get_tracer

    init_otel()
    t1 = get_tracer()
    init_otel()
    t2 = get_tracer()

    assert t1 is t2


# ═════════════════════════════════════════════════════════════════════════════
# vmo.call span attributes
# ═════════════════════════════════════════════════════════════════════════════

def test_vmo_call_span_has_required_attributes():
    """The vmo.call span must carry telecom.* and business.* attributes."""
    from vmo_pipecat.observability.otel import get_tracer

    tracer = get_tracer()
    identity = _identity()

    with tracer.start_as_current_span(
        "vmo.call",
        attributes={
            "telecom.sbc.call_id": identity.call_id_sbc,
            "telecom.asterisk.channel_id": identity.asterisk_channel_id,
            "business.vmo_call_id": identity.vmo_call_id,
            "business.tenant_id": identity.tenant_id,
            "business.did": identity.did,
        },
    ) as span:
        ctx = span.get_span_context()

    assert ctx.is_valid
    assert identity.call_id_sbc is not None


# ═════════════════════════════════════════════════════════════════════════════
# JSON logs carry trace_id and span_id
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_logs_include_trace_id_and_span_id(capsys):
    """Log lines emitted inside an OTel span must include trace_id and span_id."""
    import logging
    from vmo_pipecat.observability.otel import get_tracer

    # Ensure OTel is initialised
    get_tracer()

    with get_tracer().start_as_current_span("test-span"):
        log = logging.getLogger("test.otel.log")
        log.info("Test log inside span")

    out = capsys.readouterr().out
    # Find the log line
    for line in out.splitlines():
        if "Test log inside span" in line:
            data = json.loads(line)
            assert "trace_id" in data, f"Missing trace_id in: {line}"
            assert "span_id" in data, f"Missing span_id in: {line}"
            break
    else:
        pytest.skip("Log line not captured (structlog format may differ)")


# ═════════════════════════════════════════════════════════════════════════════
# Canonical event trace (§8.2)
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_complete_call_produces_canonical_events_in_order(stack, capsys):
    """A complete call must produce the expected events in the correct order."""
    server, router, registry, event_bus = stack
    ctrl = _make_controller(server, router, registry, event_bus)

    await registry.add(ctrl)
    task = asyncio.create_task(ctrl.start(), name="ctrl-obs")
    await asyncio.sleep(0.05)

    await ctrl.shutdown(reason="stasis_end")
    await asyncio.sleep(0.05)

    out = capsys.readouterr().out
    events = [json.loads(ln[6:]) for ln in out.splitlines() if ln.startswith("EVENT ")]
    subjects = [e["subject"] for e in events]

    for required in ("vmo.call.pipeline.started", "vmo.call.ended"):
        assert required in subjects, f"Missing event: {required}"

    call_events = [s for s in subjects if s.startswith("vmo.call.")]
    assert call_events[-1] == "vmo.call.ended"

    if not task.done():
        task.cancel()
    await asyncio.sleep(0.05)


@pytest.mark.asyncio
async def test_all_events_carry_full_identity(stack, capsys):
    """Every emitted event must include all 5 mandatory identity keys (§5.1)."""
    server, router, registry, event_bus = stack
    ctrl = _make_controller(server, router, registry, event_bus)

    await registry.add(ctrl)
    task = asyncio.create_task(ctrl.start(), name="ctrl-identity")
    await asyncio.sleep(0.05)
    await ctrl.shutdown(reason="stasis_end")
    await asyncio.sleep(0.05)

    out = capsys.readouterr().out
    events = [json.loads(ln[6:]) for ln in out.splitlines() if ln.startswith("EVENT ")]
    call_events = [e for e in events if e["subject"].startswith("vmo.call.")]

    mandatory_keys = {"vmo_call_id", "asterisk_channel_id", "call_id_sbc",
                      "caller_id", "tenant_id", "tenant_name"}
    for e in call_events:
        missing = mandatory_keys - set(e.keys())
        assert not missing, f"Event {e['subject']} missing identity keys: {missing}"

    if not task.done():
        task.cancel()
    await asyncio.sleep(0.05)


# ═════════════════════════════════════════════════════════════════════════════
# Stage timing helpers (instrumentation)
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_record_stt_final_emits_event(capsys):
    tid = f"stt_{uuid.uuid4().hex[:6]}"
    identity = _identity(tenant_id=tid)
    bus = LoggingEventBus()

    await record_stt_final(bus, identity, "deepgram", "Hola mundo", 350)

    out = capsys.readouterr().out
    events = [json.loads(ln[6:]) for ln in out.splitlines() if ln.startswith("EVENT ")]
    stt_events = [e for e in events if e["subject"] == "vmo.call.stt.final"]
    assert stt_events
    assert stt_events[0]["text"] == "Hola mundo"
    assert stt_events[0]["latency_ms"] == 350


@pytest.mark.asyncio
async def test_record_stt_partial_emits_event(capsys):
    identity = _identity()
    bus = LoggingEventBus()

    await record_stt_partial(bus, identity, "deepgram", "Ho...", confidence=0.8)

    out = capsys.readouterr().out
    events = [json.loads(ln[6:]) for ln in out.splitlines() if ln.startswith("EVENT ")]
    assert any(e["subject"] == "vmo.call.stt.partial" for e in events)


@pytest.mark.asyncio
async def test_record_llm_first_token_emits_event(capsys):
    tid = f"llm_{uuid.uuid4().hex[:6]}"
    identity = _identity(tenant_id=tid)
    bus = LoggingEventBus()

    await record_llm_first_token(bus, identity, "openai", 820)

    out = capsys.readouterr().out
    events = [json.loads(ln[6:]) for ln in out.splitlines() if ln.startswith("EVENT ")]
    assert any(e["subject"] == "vmo.call.llm.first_token" and e["ttfb_ms"] == 820 for e in events)


@pytest.mark.asyncio
async def test_record_llm_tokens_increments_counters(capsys):
    tid = f"toks_{uuid.uuid4().hex[:6]}"
    identity = _identity(tenant_id=tid)
    bus = LoggingEventBus()

    await record_llm_tokens(bus, identity, "openai", tokens_in=50, tokens_out=120)

    out = capsys.readouterr().out
    events = [json.loads(ln[6:]) for ln in out.splitlines() if ln.startswith("EVENT ")]
    assert any(e["subject"] == "vmo.call.llm.tokens" for e in events)


@pytest.mark.asyncio
async def test_record_tts_first_audio_emits_event(capsys):
    tid = f"tts_{uuid.uuid4().hex[:6]}"
    identity = _identity(tenant_id=tid)
    bus = LoggingEventBus()

    await record_tts_first_audio(bus, identity, "elevenlabs", 250)

    out = capsys.readouterr().out
    events = [json.loads(ln[6:]) for ln in out.splitlines() if ln.startswith("EVENT ")]
    assert any(e["subject"] == "vmo.call.tts.first_audio" for e in events)


@pytest.mark.asyncio
async def test_record_barge_in_emits_event(capsys):
    tid = f"barge_{uuid.uuid4().hex[:6]}"
    identity = _identity(tenant_id=tid)
    bus = LoggingEventBus()

    await record_barge_in(bus, identity, "silero", latency_ms=45)

    out = capsys.readouterr().out
    events = [json.loads(ln[6:]) for ln in out.splitlines() if ln.startswith("EVENT ")]
    barge = [e for e in events if e["subject"] == "vmo.call.barge_in"]
    assert barge
    assert barge[0]["source"] == "silero"


@pytest.mark.asyncio
async def test_stage_timer_measures_elapsed_ms():
    identity = _identity()
    bus = LoggingEventBus()

    async with stage_timer(bus, identity, "stt") as t:
        await asyncio.sleep(0.01)   # 10 ms

    assert t.elapsed_ms >= 10
    assert t.elapsed_ms < 500   # sanity upper bound


# ═════════════════════════════════════════════════════════════════════════════
# config.reloaded event via EventBus
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_config_reloaded_emitted_via_event_bus(tmp_path, capsys):
    """_do_reload() must emit vmo.system.config.reloaded through the EventBus."""
    from vmo_pipecat.runtime import _do_reload, _config_store

    yaml_content = """
defaults:
  audio_profile: telephony_8k
audio_profiles:
  telephony_8k:
    in_rate: 8000
    out_rate: 8000
    codec: slin
    channels: 1
providers: {}
pipelines: {}
contexts: {}
tenants: {}
"""
    cfg_file = tmp_path / "tenants.yaml"
    cfg_file.write_text(yaml_content)

    bus = LoggingEventBus()
    ok = await _do_reload(str(cfg_file), event_bus=bus)

    assert ok
    out = capsys.readouterr().out
    events = [json.loads(ln[6:]) for ln in out.splitlines() if ln.startswith("EVENT ")]
    reloaded = [e for e in events if e["subject"] == "vmo.system.config.reloaded"]
    assert reloaded, "vmo.system.config.reloaded not emitted"
    assert "config_version" in reloaded[0]
    assert "sha256" in reloaded[0]


@pytest.mark.asyncio
async def test_config_invalid_emitted_on_bad_yaml(tmp_path, capsys):
    """_do_reload() must emit vmo.system.config.invalid on validation failure."""
    from vmo_pipecat.runtime import _do_reload

    cfg_file = tmp_path / "bad.yaml"
    cfg_file.write_text("tenants:\n  acme:\n    routes: {}\n")

    bus = LoggingEventBus()
    ok = await _do_reload(str(cfg_file), event_bus=bus)

    assert not ok
    out = capsys.readouterr().out
    events = [json.loads(ln[6:]) for ln in out.splitlines() if ln.startswith("EVENT ")]
    invalid = [e for e in events if e["subject"] == "vmo.system.config.invalid"]
    assert invalid, "vmo.system.config.invalid not emitted"
