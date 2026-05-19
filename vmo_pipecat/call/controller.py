"""
CallController — one instance per active call.

Phase 3: transport-only echo mode when PipeCat is not installed.
Phase 4+: PipelineFactory builds the real PipeCat pipeline (STT+LLM+TTS or full-agent).

All ARI calls go through pool.client_for(identity.node_id).
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional, Any

import structlog

from .identity import CallIdentity
from ..observability.log_setup import get_logger
from ..observability.otel_metrics import record_call_started, record_call_ended
from .metrics import CallMetrics

if TYPE_CHECKING:
    from opentelemetry.trace import Span
    from ..ari.pool import ARIPool
    from ..audio.audiosocket_server import AudioSocketServer
    from ..call.router import CallRouter
    from ..call.registry import CallRegistry
    from ..events.bus import EventBus
    from ..tenancy.resolver import SessionConfig
    from ..transport.asterisk_transport import AsteriskAudioSocketTransport

logger = get_logger(__name__)


class CallController:
    """Manages one call from StasisStart to cleanup.

    Lifecycle:
      1. Created by CallLifecycle → start() spawned as async Task.
      2. bind_audio_conn() called by CallRouter after UUID handshake.
      3. on_audio() / on_dtmf() deliver real-time events from AudioSocket.
      4. shutdown() cancels everything, emits vmo.call.ended.
    """

    def __init__(
        self,
        identity: CallIdentity,
        session_config: "SessionConfig",
        bridge_id: str,
        pool: "ARIPool",
        audiosocket: "AudioSocketServer",
        transport: "AsteriskAudioSocketTransport",
        router: "CallRouter",
        registry: "CallRegistry",
        event_bus: "EventBus",
        call_span: "Span | None" = None,
    ) -> None:
        self.identity = identity
        self.session_config = session_config
        self.bridge_id = bridge_id

        self._pool = pool
        self._audiosocket = audiosocket
        self._transport = transport
        self._router = router
        self._registry = registry
        self._event_bus = event_bus
        self._call_span = call_span

        self._shutdown_event = asyncio.Event()
        self._started_at = time.monotonic()
        self._audio_channel_id: Optional[str] = None

        # Pipeline handles (set in start() if PipeCat is available)
        self._runner: Optional[Any] = None
        self._task: Optional[Any] = None
        # Phase 3 fallback: echo raw audio bytes
        self._echo_mode: bool = True
        # Per-call latency metrics
        self._call_metrics: Optional[CallMetrics] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Build PipeCat pipeline and run it (Phase 4+).

        Falls back to echo mode when pipecat-ai is not installed.
        """
        structlog.contextvars.bind_contextvars(**self.identity.asdict())
        pipeline_kind = getattr(self.session_config.pipeline, "kind", "unknown")
        record_call_started(self.identity.tenant_id, pipeline_kind)

        try:
            from ..actions.asterisk_actions import AsteriskActions
            from ..pipelines.factory import PipelineFactory
            from ..pipelines.greeting import emit_greeting

            actions = AsteriskActions(
                pool=self._pool,
                identity=self.identity,
                transfer_context=self.session_config.transfer.context,
                event_bus=self._event_bus,
            )

            self._call_metrics = CallMetrics(self.identity, self._event_bus, parent_span=self._call_span)
            self._call_metrics.set_pipeline_kind(pipeline_kind)

            runner, task = PipelineFactory.build(
                self.session_config, self._transport, actions,
                call_metrics=self._call_metrics,
            )
            self._runner = runner
            self._task = task
            self._echo_mode = False

            await self._event_bus.emit(
                "vmo.call.pipeline.started",
                self.identity,
                pipeline_kind=self.session_config.pipeline.kind,
                bridge_id=self.bridge_id,
            )

            logger.info("PipeCat pipeline starting", pipeline_kind=self.session_config.pipeline.kind)

            # Emit greeting before first user turn
            await emit_greeting(task, self.session_config)

            # Blocks until pipeline ends (EndFrame) or is cancelled
            await runner.run(task)

        except ImportError:
            # pipecat-ai not installed → Phase 3 echo fallback
            self._echo_mode = True
            logger.info("PipeCat not available — echo mode active")
            await self._event_bus.emit(
                "vmo.call.pipeline.started",
                self.identity,
                pipeline_kind="echo_dummy",
                bridge_id=self.bridge_id,
            )
            await self._shutdown_event.wait()

        except Exception as exc:
            logger.error("Pipeline error", error=str(exc), exc_info=True)
            await self._event_bus.emit(
                "vmo.call.error",
                self.identity,
                component="pipeline",
                error_type=type(exc).__name__,
                message=str(exc),
            )

        # Pipeline ended — shutdown if not already done
        if not self._shutdown_event.is_set():
            await self.shutdown(reason="pipeline_ended")

    async def bind_audio_conn(self, conn_id: str) -> None:
        """Called by CallRouter once the AudioSocket UUID handshake succeeds."""
        self._transport.bind(conn_id)
        logger.info("AudioSocket bound", conn_id=conn_id)
        await self._event_bus.emit(
            "vmo.call.audio.connected",
            self.identity,
            conn_id=conn_id,
        )

    # ------------------------------------------------------------------
    # Audio / DTMF / disconnect
    # ------------------------------------------------------------------

    _audio_rx_count: int = 0  # debug counter
    _max_rms_seen: float = 0.0  # Para detectar si hay voz real

    async def on_audio(self, audio: bytes) -> None:
        self._audio_rx_count += 1
        # Calcular energía RMS del frame para detectar voz vs silencio
        if self._audio_rx_count % 50 == 0:
            try:
                import struct
                samples = struct.unpack(f"<{len(audio)//2}h", audio)
                rms = (sum(s*s for s in samples) / len(samples)) ** 0.5
                if rms > self._max_rms_seen:
                    self._max_rms_seen = rms
                logger.debug("audio energy", count=self._audio_rx_count,
                             rms=round(rms), max_rms=round(self._max_rms_seen))
            except Exception:
                pass
        if self._echo_mode:
            await self._transport.output().write_raw_audio_frames(audio)
        else:
            await self._transport.push_audio(audio)

    async def on_dtmf(self, digit: str) -> None:
        """Inject a DTMF digit as DTMFFrame into the active pipeline."""
        logger.info("DTMF received", digit=digit)
        if self._task is not None and not self._echo_mode:
            try:
                from ..events.from_asterisk import DTMFFrame
                await self._task.queue_frame(
                    DTMFFrame(digit=digit, channel_id=self.identity.asterisk_channel_id)
                )
            except Exception as exc:
                logger.debug("Failed to queue DTMFFrame", error=str(exc))

    async def on_talking_started(self) -> None:
        """Translate ChannelTalkingStarted → UserStartedSpeakingFrame.

        Only active when pipeline.vad == 'asterisk_talk_detect' (§4.2).
        Default: Silero VAD handles this natively — ignore the ARI event.
        """
        vad_kind = getattr(self.session_config.pipeline, "vad", "silero")
        if vad_kind != "asterisk_talk_detect":
            return
        if self._task is not None and not self._echo_mode:
            try:
                from ..events.from_asterisk import make_user_started_speaking
                await self._task.queue_frame(make_user_started_speaking())
            except Exception as exc:
                logger.debug("Failed to queue UserStartedSpeakingFrame", error=str(exc))

    async def on_talking_finished(self) -> None:
        """Translate ChannelTalkingFinished → UserStoppedSpeakingFrame (§4.2)."""
        vad_kind = getattr(self.session_config.pipeline, "vad", "silero")
        if vad_kind != "asterisk_talk_detect":
            return
        if self._task is not None and not self._echo_mode:
            try:
                from ..events.from_asterisk import make_user_stopped_speaking
                await self._task.queue_frame(make_user_stopped_speaking())
            except Exception as exc:
                logger.debug("Failed to queue UserStoppedSpeakingFrame", error=str(exc))

    async def on_disconnect(self) -> None:
        """AudioSocket connection dropped."""
        await self.shutdown(reason="audiosocket_disconnect")

    # ------------------------------------------------------------------
    # Shutdown (idempotent)
    # ------------------------------------------------------------------

    async def shutdown(self, reason: str = "normal") -> None:
        if self._shutdown_event.is_set():
            return
        self._shutdown_event.set()

        structlog.contextvars.bind_contextvars(**self.identity.asdict())
        logger.info("CallController shutting down", reason=reason)

        # Cancel PipeCat pipeline if running
        if self._task is not None:
            try:
                await self._task.cancel()
            except Exception as exc:
                logger.debug("Pipeline cancel error", error=str(exc))

        # Remove from routing maps
        self._router.remove(self)
        await self._registry.remove(self)

        ari = self._pool.client_for(self.identity.node_id)

        # Always hang up the AudioSocket channel
        if self._audio_channel_id:
            try:
                await ari.hangup_channel(self._audio_channel_id)
            except Exception as exc:
                logger.debug("AudioSocket hangup failed", error=str(exc))

        # Hangup the caller channel (unless Asterisk already did it)
        if reason not in ("hangup", "stasis_end"):
            try:
                await ari.hangup_channel(self.identity.asterisk_channel_id)
            except Exception as exc:
                logger.warning("Hangup failed during shutdown", error=str(exc))

        duration_ms = int((time.monotonic() - self._started_at) * 1000)
        pipeline_kind = getattr(self.session_config.pipeline, "kind", "unknown")
        record_call_ended(
            tenant_id=self.identity.tenant_id,
            pipeline_kind=pipeline_kind,
            outcome=reason,
            duration_s=duration_ms / 1000.0,
        )
        summary = self._call_metrics.summary() if self._call_metrics else {}
        await self._event_bus.emit(
            "vmo.call.ended",
            self.identity,
            outcome=reason,
            duration_ms=duration_ms,
            summary=summary,
        )
        logger.info("CallController stopped", reason=reason, duration_ms=duration_ms, **summary)

        # Close OTel call span
        if self._call_span is not None:
            self._call_span.end()
