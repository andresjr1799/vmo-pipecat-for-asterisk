"""Deepgram Voice Agent full-agent provider (custom, replaces missing Pipecat service).

Conecta via WebSocket a wss://agent.deepgram.com/v1/agent/converse.
Maneja STT + LLM + TTS + VAD + barge-in nativamente en una sola conexion.
Basado en el patron de AVA ai voice for asterisk.
"""

from __future__ import annotations

import asyncio
import audioop
import contextlib
import json
import time
from typing import TYPE_CHECKING, Any, Optional

from ..observability.log_setup import get_logger

logger = get_logger(__name__)

try:
    import websockets
    from websockets.asyncio.client import connect as ws_connect
    from websockets import exceptions as ws_exceptions
    from websockets.protocol import State as WsState
except ImportError:
    websockets = None  # type: ignore[assignment]
    ws_exceptions = None  # type: ignore[assignment]
    WsState = None  # type: ignore[assignment]

from pipecat.frames.frames import (
    CancelFrame, EndFrame, Frame, InputAudioRawFrame,
    InterruptionFrame, StartFrame, TTSAudioRawFrame,
    TTSStartedFrame, TTSStoppedFrame, UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame, TranscriptionFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.utils.time import time_now_iso8601

if TYPE_CHECKING:
    from ..config.models import DeepgramVoiceAgentProviderCfg, AudioProfileCfg


def build_service(
    resolved: "DeepgramVoiceAgentProviderCfg",
    audio_profile: "AudioProfileCfg",
    *,
    context_prompt: str = "",
    context_greeting: str = "",
) -> Any:
    """Factory for PipelineFactory. Returns a DeepgramVoiceAgentService."""
    params = dict(resolved.params)
    return DeepgramVoiceAgentService(
        api_key=resolved.api_key,
        sample_rate=audio_profile.in_rate,
        input_encoding="mulaw",
        output_sample_rate=audio_profile.out_rate,
        instructions=context_prompt or params.pop("instructions", ""),
        greeting=context_greeting or params.pop("greeting", ""),
        think_provider=params.pop("think_provider", "open_ai"),
        think_model=params.pop("think_model", "gpt-4o-mini"),
        **params,
    )


class DeepgramVoiceAgentService(FrameProcessor):
    """Full-agent: STT + LLM + TTS + VAD via Deepgram Voice Agent WebSocket."""

    def __init__(
        self,
        *,
        api_key: str,
        sample_rate: int = 8000,
        input_encoding: str = "mulaw",
        output_sample_rate: int = 16000,
        model: str = "nova-3",
        tts_model: str = "aura-2-celeste-es",
        language: str = "es",
        instructions: str = "",
        greeting: str = "",
        think_provider: str = "open_ai",
        think_model: str = "gpt-4o-mini",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._api_key = api_key
        self._sample_rate = sample_rate
        self._input_encoding = input_encoding
        self._output_sample_rate = output_sample_rate
        self._model = model
        self._tts_model = tts_model
        self._language = language
        self._instructions = instructions
        self._greeting = greeting
        self._think_provider = think_provider
        self._think_model = think_model
        self._think_temperature = kwargs.pop("think_temperature", 0.7)

        self._ws: Any = None
        self._receive_task: Optional[asyncio.Task] = None
        self._closing = False
        self._audio_burst = False

        # Local energy VAD — requires sustained speech to avoid coughs
        self._vad_speech_count = 0
        self._vad_silence_count = 0
        self._vad_active = False
        self._vad_rms_threshold = 800
        self._vad_start_frames = 10      # 200ms continuous speech to trigger
        self._vad_stop_frames = 30       # 600ms silence to reset

        # OTel: agent_e2e_turn_around tracking
        self._agent_e2e_span: Any = None
        self._agent_e2e_start: float = 0.0

    async def _connect(self):
        if self._ws:
            return
        url = f"wss://agent.deepgram.com/v1/agent/converse"
        headers = {"Authorization": f"Token {self._api_key}"}
        try:
            self._ws = await ws_connect(url, additional_headers=headers, max_size=16 * 1024 * 1024)
            logger.info(f"{self}: Connected to Deepgram Voice Agent")
            # Send Settings
            settings = {
                "type": "Settings",
                "audio": {
                    "input": {"encoding": self._input_encoding, "sample_rate": self._sample_rate},
                    "output": {"encoding": "linear16", "sample_rate": self._output_sample_rate, "container": "none"},
                },
                "agent": {
                    "listen": {"provider": {"type": "deepgram", "model": self._model}},
                    "speak": {"provider": {"type": "deepgram", "model": self._tts_model}},
                    "think": {
                        "provider": {
                            "type": self._think_provider,
                            "model": self._think_model,
                            "temperature": self._think_temperature,
                        },
                        "prompt": self._instructions,
                    },
                    "language": self._language,
                    "greeting": self._greeting,
                },
            }
            await self._ws.send(json.dumps(settings))
            self._receive_task = self.create_task(self._receive_loop())
            self._audio_burst = False
        except Exception as e:
            logger.error(f"{self}: Connection failed: {e}")

    async def _disconnect(self):
        self._closing = True
        if self._receive_task:
            self._receive_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._receive_task
            self._receive_task = None
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
        self._closing = False

    async def _receive_loop(self):
        try:
            async for message in self._ws:
                if self._closing:
                    break
                if isinstance(message, bytes):
                    frame = TTSAudioRawFrame(message, self._output_sample_rate, 1)
                    if not self._audio_burst:
                        self._audio_burst = True
                        # End agent_e2e_turn_around span
                        if self._agent_e2e_span is not None and self._agent_e2e_start > 0:
                            duration_ms = (time.monotonic() - self._agent_e2e_start) * 1000
                            self._agent_e2e_span.set_attribute("duration_ms", duration_ms)
                            self._agent_e2e_span.end()
                            self._agent_e2e_span = None
                            self._agent_e2e_start = 0.0
                            try:
                                from ..observability.otel_metrics import record_agent_e2e_turn_around
                                record_agent_e2e_turn_around("unknown", "full_agent", duration_ms)
                            except Exception:
                                pass
                        await self.push_frame(TTSStartedFrame())
                    await self.push_frame(frame)
                elif isinstance(message, str):
                    await self._handle_json(message)
        except ws_exceptions.ConnectionClosed:
            logger.info(f"{self}: WebSocket closed")
        except asyncio.CancelledError:
            pass
        finally:
            if self._audio_burst:
                self._audio_burst = False
                await self.push_frame(TTSStoppedFrame())

    async def _handle_json(self, raw: str):
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return
        t = msg.get("type", "")
        # Log all events to debug VAD/interruption behavior
        if t not in ("Metadata", "Flushed", "Cleared", "Welcome", "SettingsApplied"):
            logger.info(f"{self}: Deepgram event: type={t} keys={list(msg.keys())}")
        if t == "UserStartedSpeaking":
            await self.push_frame(UserStartedSpeakingFrame(), FrameDirection.UPSTREAM)
            await self.push_frame(InterruptionFrame())
        elif t == "UserStoppedSpeaking":
            await self.push_frame(UserStoppedSpeakingFrame(), FrameDirection.UPSTREAM)
            self._agent_e2e_start = time.monotonic()
            try:
                from ..observability.otel import get_tracer
                self._agent_e2e_span = get_tracer().start_span("agent_e2e_turn_around")
            except Exception:
                self._agent_e2e_span = None
        elif t == "ConversationText":
            role = msg.get("role", "")
            text = msg.get("content", "") or msg.get("text", "")
            if text and role == "user":
                await self.push_frame(TranscriptionFrame(text, "", time_now_iso8601()))
        elif t == "SettingsApplied":
            logger.info(f"{self}: Settings applied by Deepgram")
        elif t in ("Welcome", "Metadata", "Flushed", "Cleared"):
            pass
        elif t in ("Error", "Warning"):
            logger.warning(f"{self}: Deepgram {t}: {msg}")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, InputAudioRawFrame):
            # Local energy VAD — requires sustained speech (200ms) to reject coughs
            rms = audioop.rms(frame.audio, 2)
            if rms > self._vad_rms_threshold:
                self._vad_speech_count += 1
                self._vad_silence_count = 0
                if self._vad_speech_count >= self._vad_start_frames and not self._vad_active:
                    self._vad_active = True
                    await self.push_frame(InterruptionFrame())
            else:
                self._vad_silence_count += 1
                self._vad_speech_count = 0
                if self._vad_silence_count >= self._vad_stop_frames and self._vad_active:
                    self._vad_active = False

            # Send to Deepgram
            if self._ws and self._ws.state == WsState.OPEN:
                payload = audioop.lin2ulaw(frame.audio, 2) if self._input_encoding == "mulaw" else frame.audio
                await self._ws.send(payload)
            return

        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            await self._connect()
            await self.push_frame(frame, direction)
        elif isinstance(frame, EndFrame):
            await self._disconnect()
            await self.push_frame(frame, direction)
        elif isinstance(frame, CancelFrame):
            await self._disconnect()
            await self.push_frame(frame, direction)
        elif isinstance(frame, InterruptionFrame):
            await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)

    @property
    def sample_rate(self) -> int:
        return self._output_sample_rate
