"""
Instrumentation helpers for stage timing and latency recording (Phase 9).

Used by PipeCat pipeline hooks and controller callbacks to emit canonical
events and record Prometheus histograms for each voice pipeline stage.

Usage example (when PipeCat is available and a processor fires):
    async with stage_timer(event_bus, identity, "stt") as t:
        transcript = await stt_service.transcribe(audio)
    record_stt_final(event_bus, identity, "deepgram", t.elapsed_ms)
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, AsyncIterator, Optional

from ..observability.log_setup import get_logger

if TYPE_CHECKING:
    from ..call.identity import CallIdentity
    from ..events.bus import EventBus

logger = get_logger(__name__)


# ── Timing context manager ─────────────────────────────────────────────────────

@dataclass
class StageTimer:
    """Records wall-clock time for a pipeline stage."""
    stage: str
    start: float = field(default_factory=time.monotonic)
    _end: Optional[float] = field(default=None, repr=False)

    def stop(self) -> None:
        self._end = time.monotonic()

    @property
    def elapsed_ms(self) -> int:
        end = self._end if self._end is not None else time.monotonic()
        return int((end - self.start) * 1000)


@asynccontextmanager
async def stage_timer(
    event_bus: "EventBus",
    identity: "CallIdentity",
    stage: str,
) -> AsyncIterator[StageTimer]:
    """Async context manager that times a pipeline stage and logs it."""
    t = StageTimer(stage=stage)
    try:
        yield t
    finally:
        t.stop()
        logger.debug("Stage timing", stage=stage, latency_ms=t.elapsed_ms)


# ── Per-stage event + metric helpers ──────────────────────────────────────────

async def record_stt_final(
    event_bus: "EventBus",
    identity: "CallIdentity",
    provider: str,
    text: str,
    latency_ms: int,
) -> None:
    """Emit vmo.call.stt.final + record stt_latency histogram via OTel."""
    from ..observability.otel_metrics import record_stt_latency
    record_stt_latency(identity.tenant_id, provider, latency_ms)
    await event_bus.emit(
        "vmo.call.stt.final",
        identity,
        provider=provider,
        text=text,
        latency_ms=latency_ms,
    )


async def record_stt_partial(
    event_bus: "EventBus",
    identity: "CallIdentity",
    provider: str,
    text: str,
    confidence: float = 0.0,
) -> None:
    """Emit vmo.call.stt.partial (no histogram — partials are frequent)."""
    await event_bus.emit(
        "vmo.call.stt.partial",
        identity,
        provider=provider,
        text=text,
        confidence=confidence,
    )


async def record_llm_first_token(
    event_bus: "EventBus",
    identity: "CallIdentity",
    provider: str,
    ttfb_ms: int,
) -> None:
    """Emit vmo.call.llm.first_token + record llm_ttfb histogram via OTel."""
    from ..observability.otel_metrics import record_llm_ttfb
    record_llm_ttfb(identity.tenant_id, provider, ttfb_ms)
    await event_bus.emit(
        "vmo.call.llm.first_token",
        identity,
        provider=provider,
        ttfb_ms=ttfb_ms,
    )


async def record_llm_tokens(
    event_bus: "EventBus",
    identity: "CallIdentity",
    provider: str,
    tokens_in: int,
    tokens_out: int,
) -> None:
    """Emit vmo.call.llm.tokens + increment llm_tokens counters via OTel."""
    from ..observability.otel_metrics import record_llm_tokens as _record
    _record(identity.tenant_id, provider, tokens_in, tokens_out)
    await event_bus.emit(
        "vmo.call.llm.tokens",
        identity,
        provider=provider,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
    )


async def record_tts_first_audio(
    event_bus: "EventBus",
    identity: "CallIdentity",
    provider: str,
    latency_ms: int,
) -> None:
    """Emit vmo.call.tts.first_audio + record tts_first_audio histogram via OTel."""
    from ..observability.otel_metrics import record_tts_first_audio as _record
    _record(identity.tenant_id, provider, latency_ms)
    await event_bus.emit(
        "vmo.call.tts.first_audio",
        identity,
        provider=provider,
        latency_ms=latency_ms,
    )


async def record_turn_response(
    event_bus: "EventBus",
    identity: "CallIdentity",
    pipeline_kind: str,
    turn_ms: int,
) -> None:
    """Emit vmo.call turn timing + record turn_response histogram via OTel."""
    from ..observability.otel_metrics import record_turn_response as _record
    _record(identity.tenant_id, pipeline_kind, turn_ms)


async def record_barge_in(
    event_bus: "EventBus",
    identity: "CallIdentity",
    source: str,
    latency_ms: int = 0,
) -> None:
    """Emit vmo.call.barge_in + increment barge_in counter via OTel."""
    from ..observability.otel_metrics import record_barge_in as _record
    _record(identity.tenant_id, source)
    await event_bus.emit(
        "vmo.call.barge_in",
        identity,
        source=source,
        latency_ms=latency_ms,
    )
