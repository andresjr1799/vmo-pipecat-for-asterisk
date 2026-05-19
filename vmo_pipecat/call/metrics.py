"""CallMetrics — per-call latency tracking for STT/LLM/TTS stages.

Acumula latencias por turno y emite el evento vmo.call.ended
con un summary que incluye p50 de cada etapa (arquitectura §8.2).

OTel: cada stage crea un child span del span padre vmo.call.
Las metricas se registran via otel_metrics (OTLP push).
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Optional

from ..observability.log_setup import get_logger
from ..observability.otel_metrics import (
    record_stt_latency,
    record_llm_ttfb,
    record_tts_first_audio,
    record_turn_response,
)

if TYPE_CHECKING:
    from opentelemetry.trace import Span
    from .identity import CallIdentity
    from ..events.bus import EventBus

logger = get_logger(__name__)


def _p50(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[len(s) // 2]


class CallMetrics:
    """Tracks STT/LLM/TTS latencies across all turns of a single call."""

    def __init__(self, identity: "CallIdentity", event_bus: "EventBus", parent_span: "Span | None" = None):
        self._identity = identity
        self._event_bus = event_bus
        self._tenant_id = identity.tenant_id or "unknown"
        self._parent_span = parent_span

        self.stt_latencies: list[float] = []
        self.llm_ttfbs: list[float] = []
        self.tts_latencies: list[float] = []
        self.turn_latencies: list[float] = []   # stt_final → tts_first_audio
        self.barge_ins_count: int = 0
        self.errors_count: int = 0

        # Per-turn timing state
        self._turn_stt_start: float = 0.0
        self._turn_stt_end: Optional[float] = None
        self._turn_llm_start: Optional[float] = None
        self._turn_tts_first: Optional[float] = None
        self._provider_stt: str = ""
        self._provider_llm: str = ""
        self._provider_tts: str = ""
        self._pipeline_kind: str = ""

        # Active OTel child spans for current turn
        self._stt_span: Optional["Span"] = None
        self._llm_span: Optional["Span"] = None

    # ── Stage hooks ─────────────────────────────────────────────────────────

    def set_providers(self, stt: str, llm: str, tts: str) -> None:
        self._provider_stt = stt
        self._provider_llm = llm
        self._provider_tts = tts

    def set_pipeline_kind(self, kind: str) -> None:
        self._pipeline_kind = kind

    def stt_start(self) -> None:
        self._turn_stt_start = time.monotonic()
        self._turn_stt_end = None
        self._turn_llm_start = None
        self._turn_tts_first = None
        # Start STT child span with parent context
        if self._parent_span is not None:
            try:
                from opentelemetry import context as otel_ctx
                from opentelemetry import trace as otel_trace
                from ..observability.otel import get_tracer
                parent_ctx = otel_trace.set_span_in_context(self._parent_span)
                self._stt_span = get_tracer().start_span(
                    "stt_latency",
                    attributes={"provider": self._provider_stt},
                    context=parent_ctx,
                )
            except Exception:
                self._stt_span = None

    def stt_final(self, text: str = "") -> None:
        if not self._turn_stt_end:
            self._turn_stt_end = time.monotonic()
            latency_ms = (self._turn_stt_end - self._turn_stt_start) * 1000
            self.stt_latencies.append(latency_ms)
            record_stt_latency(self._tenant_id, self._provider_stt, latency_ms)
            logger.debug("STT final", provider=self._provider_stt, text=text[:80] if text else "", latency_ms=int(latency_ms))

        # Close STT span + start LLM span
        if self._stt_span is not None:
            self._stt_span.end()
            self._stt_span = None
        if self._parent_span is not None and self._turn_stt_end:
            try:
                from opentelemetry import context as otel_ctx
                from opentelemetry import trace as otel_trace
                from ..observability.otel import get_tracer
                parent_ctx = otel_trace.set_span_in_context(self._parent_span)
                self._llm_span = get_tracer().start_span(
                    "llm_ttfb",
                    attributes={"provider": self._provider_llm},
                    context=parent_ctx,
                )
            except Exception:
                self._llm_span = None

    def llm_first_token(self) -> None:
        if not self._turn_llm_start and self._turn_stt_end:
            self._turn_llm_start = time.monotonic()
            # Compute LLM TTFB: time from STT final to first LLM token
            ttfb_ms = (self._turn_llm_start - self._turn_stt_end) * 1000
            self.llm_ttfbs.append(ttfb_ms)
            record_llm_ttfb(self._tenant_id, self._provider_llm, ttfb_ms)
            logger.debug("LLM first token", provider=self._provider_llm, ttfb_ms=int(ttfb_ms))

    def tts_first_audio(self) -> None:
        if not self._turn_tts_first and self._turn_llm_start:
            self._turn_tts_first = time.monotonic()
            # TTS latency: desde que el LLM empezo a generar hasta que hay audio
            tts_ms = (self._turn_tts_first - self._turn_llm_start) * 1000
            self.tts_latencies.append(tts_ms)
            record_tts_first_audio(self._tenant_id, self._provider_tts, tts_ms)

            # Close LLM span
            if self._llm_span is not None:
                self._llm_span.end()
                self._llm_span = None

            # Record TTS first audio span
            if self._parent_span is not None:
                try:
                    from opentelemetry import context as otel_ctx
                    from opentelemetry import trace as otel_trace
                    from ..observability.otel import get_tracer
                    parent_ctx = otel_trace.set_span_in_context(self._parent_span)
                    tts_span = get_tracer().start_span(
                        "tts_ttfa",
                        attributes={"provider": self._provider_tts, "latency_ms": tts_ms},
                        context=parent_ctx,
                    )
                    tts_span.end()
                except Exception:
                    pass

            # Turn response: stt_final → tts_first_audio
            if self._turn_stt_end:
                turn_ms = (self._turn_tts_first - self._turn_stt_end) * 1000
                self.turn_latencies.append(turn_ms)
                record_turn_response(self._tenant_id, self._pipeline_kind or "modular", turn_ms)

            logger.debug("TTS first audio", provider=self._provider_tts, tts_ms=int(tts_ms))

    def record_barge_in(self) -> None:
        self.barge_ins_count += 1

    def record_error(self) -> None:
        self.errors_count += 1

    # ── Summary for vmo.call.ended ──────────────────────────────────────────

    def summary(self) -> dict:
        return {
            "stt_p50_ms": int(_p50(self.stt_latencies)),
            "stt_avg_ms": int(sum(self.stt_latencies) / len(self.stt_latencies)) if self.stt_latencies else 0,
            "stt_count": len(self.stt_latencies),
            "llm_ttfb_p50_ms": int(_p50(self.llm_ttfbs)),
            "llm_ttfb_avg_ms": int(sum(self.llm_ttfbs) / len(self.llm_ttfbs)) if self.llm_ttfbs else 0,
            "tts_first_audio_p50_ms": int(_p50(self.tts_latencies)),
            "tts_first_audio_avg_ms": int(sum(self.tts_latencies) / len(self.tts_latencies)) if self.tts_latencies else 0,
            "turn_response_p50_ms": int(_p50(self.turn_latencies)),
            "barge_ins_count": self.barge_ins_count,
            "errors_count": self.errors_count,
        }
