"""
OpenTelemetry metric definitions for vmo-pipecat (§8.5 → OTel).

Replaces prometheus_client with OTel Metrics API. All metrics are created
via get_meter() and exported via OTLP gRPC to otel-collector.

Cardinality note: tenant_id is bounded (~tens/hundreds in V1). Never label
with vmo_call_id or asterisk_channel_id (unbounded cardinality).
"""

from __future__ import annotations

from typing import Optional

from opentelemetry.metrics import Counter, Histogram, UpDownCounter, Meter

from .otel import get_meter

_meter: Optional[Meter] = None


def _m() -> Meter:
    global _meter
    if _meter is None:
        _meter = get_meter()
    return _meter


# ── Counters ───────────────────────────────────────────────────────────────────

_calls_started: Optional[Counter] = None
_calls_ended: Optional[Counter] = None
_transfer: Optional[Counter] = None
_tool_call: Optional[Counter] = None
_barge_in: Optional[Counter] = None
_audiosocket_disconnect: Optional[Counter] = None
_ari_reconnect: Optional[Counter] = None
_config_reload: Optional[Counter] = None
_llm_tokens: Optional[Counter] = None
_agentic_request_published: Optional[Counter] = None
_agentic_tool_calls: Optional[Counter] = None
_agentic_timeout: Optional[Counter] = None
_agentic_publish_errors: Optional[Counter] = None

# ── Gauges ─────────────────────────────────────────────────────────────────────

_calls_active: Optional[UpDownCounter] = None
_ari_node_connected: Optional[UpDownCounter] = None
_audiosocket_active_connections: Optional[UpDownCounter] = None

# ── Histograms ─────────────────────────────────────────────────────────────────

_call_duration_seconds: Optional[Histogram] = None
_stt_latency_ms: Optional[Histogram] = None
_llm_ttfb_ms: Optional[Histogram] = None
_tts_first_audio_ms: Optional[Histogram] = None
_turn_response_ms: Optional[Histogram] = None
_agent_e2e_turn_around_ms: Optional[Histogram] = None
_agentic_first_token_ms: Optional[Histogram] = None
_agentic_turn_total_ms: Optional[Histogram] = None

# ── AudioSocket metrics (defined here for centralisation) ──────────────────────

_audiosocket_rx_bytes: Optional[Counter] = None
_audiosocket_tx_bytes: Optional[Counter] = None


# ── Lazy initialisation helpers ────────────────────────────────────────────────

def _counter(name: str, description: str, unit: str = "1") -> Counter:
    return _m().create_counter(name, description=description, unit=unit)


def _histogram(name: str, description: str, unit: str = "ms") -> Histogram:
    return _m().create_histogram(name, unit=unit, description=description)


def _updown(name: str, description: str, unit: str = "1") -> UpDownCounter:
    return _m().create_up_down_counter(name, description=description, unit=unit)


# ── Init all metrics (idempotent) ──────────────────────────────────────────────

def init_metrics() -> None:
    """Create all OTel metric instruments. Safe to call multiple times."""
    global _calls_started, _calls_ended, _transfer, _tool_call, _barge_in
    global _audiosocket_disconnect, _ari_reconnect, _config_reload, _llm_tokens
    global _agentic_request_published, _agentic_tool_calls, _agentic_timeout
    global _agentic_publish_errors
    global _calls_active, _ari_node_connected, _audiosocket_active_connections
    global _call_duration_seconds, _stt_latency_ms, _llm_ttfb_ms, _tts_first_audio_ms
    global _turn_response_ms, _agent_e2e_turn_around_ms
    global _agentic_first_token_ms, _agentic_turn_total_ms
    global _audiosocket_rx_bytes, _audiosocket_tx_bytes

    if _calls_started is not None:
        return

    # Counters
    _calls_started = _counter("vmo.calls.started", "Total calls received")
    _calls_ended = _counter("vmo.calls.ended", "Total calls ended by outcome")
    _transfer = _counter("vmo.transfer", "Call transfer attempts by result")
    _tool_call = _counter("vmo.tool_call", "LLM tool calls by name and result")
    _barge_in = _counter("vmo.barge_in", "Barge-in events detected")
    _audiosocket_disconnect = _counter("vmo.audiosocket.disconnect", "AudioSocket disconnects by reason")
    _ari_reconnect = _counter("vmo.ari.reconnect", "ARI WebSocket reconnect attempts")
    _config_reload = _counter("vmo.config.reload", "Config hot-reload attempts")
    _llm_tokens = _counter("vmo.llm.tokens", "LLM tokens processed")
    _agentic_request_published = _counter("vmo.agentic.request_published", "Agentic bus requests published")
    _agentic_tool_calls = _counter("vmo.agentic.tool_calls", "Tool calls from agentic backend")
    _agentic_timeout = _counter("vmo.agentic.timeout", "Agentic bus timeouts")
    _agentic_publish_errors = _counter("vmo.agentic.publish_errors", "Agentic bus publish errors")
    _audiosocket_rx_bytes = _counter("vmo.audiosocket.rx_bytes", "AudioSocket bytes received", unit="By")
    _audiosocket_tx_bytes = _counter("vmo.audiosocket.tx_bytes", "AudioSocket bytes sent", unit="By")

    # UpDownCounters (Gauge-like)
    _calls_active = _updown("vmo.calls.active", "Currently active calls")
    _ari_node_connected = _updown("vmo.ari.node_connected", "ARI node connection state")
    _audiosocket_active_connections = _updown("vmo.audiosocket.active_connections", "Active AudioSocket TCP connections")

    # Histograms
    _call_duration_seconds = _m().create_histogram(
        "vmo.call.duration", unit="s", description="Call duration")
    _stt_latency_ms = _histogram("vmo.stt.latency", "STT final transcript latency")
    _llm_ttfb_ms = _histogram("vmo.llm.ttfb", "LLM time-to-first-token")
    _tts_first_audio_ms = _histogram("vmo.tts.first_audio", "TTS time-to-first-audio")
    _turn_response_ms = _histogram("vmo.turn.response", "End-to-end turn response time")
    _agent_e2e_turn_around_ms = _histogram("vmo.agent.e2e_turn_around", "Full-agent end-to-end turn around")
    _agentic_first_token_ms = _histogram("vmo.agentic.first_token", "Agentic bus time-to-first-token")
    _agentic_turn_total_ms = _histogram("vmo.agentic.turn_total", "Agentic bus total turn latency")


# ── Recording helpers ──────────────────────────────────────────────────────────

def record_call_started(tenant_id: str, pipeline_kind: str) -> None:
    assert _calls_started and _calls_active
    _calls_started.add(1, {"tenant_id": tenant_id, "pipeline_kind": pipeline_kind})
    _calls_active.add(1, {"tenant_id": tenant_id})


def record_call_ended(tenant_id: str, pipeline_kind: str, outcome: str, duration_s: float) -> None:
    assert _calls_ended and _calls_active and _call_duration_seconds
    _calls_ended.add(1, {"tenant_id": tenant_id, "pipeline_kind": pipeline_kind, "outcome": outcome})
    _calls_active.add(-1, {"tenant_id": tenant_id})
    _call_duration_seconds.record(duration_s, {"tenant_id": tenant_id, "pipeline_kind": pipeline_kind})


def record_transfer(tenant_id: str, result: str) -> None:
    assert _transfer
    _transfer.add(1, {"tenant_id": tenant_id, "result": result})


def record_tool_call(tenant_id: str, tool_name: str, result: str) -> None:
    assert _tool_call
    _tool_call.add(1, {"tenant_id": tenant_id, "tool_name": tool_name, "result": result})


def record_stt_latency(tenant_id: str, provider: str, latency_ms: float) -> None:
    assert _stt_latency_ms
    _stt_latency_ms.record(latency_ms, {"tenant_id": tenant_id, "provider": provider})


def record_llm_ttfb(tenant_id: str, provider: str, ttfb_ms: float) -> None:
    assert _llm_ttfb_ms
    _llm_ttfb_ms.record(ttfb_ms, {"tenant_id": tenant_id, "provider": provider})


def record_tts_first_audio(tenant_id: str, provider: str, latency_ms: float) -> None:
    assert _tts_first_audio_ms
    _tts_first_audio_ms.record(latency_ms, {"tenant_id": tenant_id, "provider": provider})


def record_turn_response(tenant_id: str, pipeline_kind: str, turn_ms: float) -> None:
    assert _turn_response_ms
    _turn_response_ms.record(turn_ms, {"tenant_id": tenant_id, "pipeline_kind": pipeline_kind})


def record_agent_e2e_turn_around(tenant_id: str, pipeline_kind: str, turn_ms: float) -> None:
    assert _agent_e2e_turn_around_ms
    _agent_e2e_turn_around_ms.record(turn_ms, {"tenant_id": tenant_id, "pipeline_kind": pipeline_kind})


def record_barge_in(tenant_id: str, source: str) -> None:
    assert _barge_in
    _barge_in.add(1, {"tenant_id": tenant_id, "source": source})


def record_config_reload(result: str) -> None:
    assert _config_reload
    _config_reload.add(1, {"result": result})


def record_ari_reconnect(node_id: str) -> None:
    assert _ari_reconnect
    _ari_reconnect.add(1, {"node_id": node_id})


def set_ari_node_connected(node_id: str, connected: bool) -> None:
    assert _ari_node_connected
    _ari_node_connected.add(1 if connected else -1, {"node_id": node_id})


def record_llm_tokens(tenant_id: str, provider: str, tokens_in: int, tokens_out: int) -> None:
    assert _llm_tokens
    if tokens_in:
        _llm_tokens.add(tokens_in, {"tenant_id": tenant_id, "provider": provider, "direction": "in"})
    if tokens_out:
        _llm_tokens.add(tokens_out, {"tenant_id": tenant_id, "provider": provider, "direction": "out"})


def record_audiosocket_disconnect(reason: str) -> None:
    assert _audiosocket_disconnect
    _audiosocket_disconnect.add(1, {"reason": reason})


# ── AudioSocket specific ───────────────────────────────────────────────────────

def audiosocket_conn_inc() -> None:
    assert _audiosocket_active_connections
    _audiosocket_active_connections.add(1)


def audiosocket_conn_dec() -> None:
    assert _audiosocket_active_connections
    _audiosocket_active_connections.add(-1)


def audiosocket_rx_bytes_inc(count: int) -> None:
    assert _audiosocket_rx_bytes
    _audiosocket_rx_bytes.add(count)


def audiosocket_tx_bytes_inc(count: int) -> None:
    assert _audiosocket_tx_bytes
    _audiosocket_tx_bytes.add(count)


# ── Agentic bus ────────────────────────────────────────────────────────────────

def record_agentic_request(tenant_id: str, transport: str, result: str) -> None:
    assert _agentic_request_published
    _agentic_request_published.add(1, {"tenant_id": tenant_id, "transport": transport, "result": result})


def record_agentic_first_token(tenant_id: str, transport: str, latency_ms: float) -> None:
    assert _agentic_first_token_ms
    _agentic_first_token_ms.record(latency_ms, {"tenant_id": tenant_id, "transport": transport})


def record_agentic_turn(tenant_id: str, transport: str, total_ms: float) -> None:
    assert _agentic_turn_total_ms
    _agentic_turn_total_ms.record(total_ms, {"tenant_id": tenant_id, "transport": transport})


def record_agentic_tool_call(tenant_id: str, tool_name: str) -> None:
    assert _agentic_tool_calls
    _agentic_tool_calls.add(1, {"tenant_id": tenant_id, "tool_name": tool_name})


def record_agentic_timeout(tenant_id: str, phase: str) -> None:
    assert _agentic_timeout
    _agentic_timeout.add(1, {"tenant_id": tenant_id, "phase": phase})


def record_agentic_publish_error(tenant_id: str, transport: str, error_type: str) -> None:
    assert _agentic_publish_errors
    _agentic_publish_errors.add(1, {"tenant_id": tenant_id, "transport": transport, "error_type": error_type})
