"""ElevenLabs Conversational AI full-agent provider (custom).

Conecta via WebSocket a wss://api.elevenlabs.io/v1/convai/conversation.
Maneja STT + LLM + TTS + VAD + barge-in nativamente en una sola conexion.
Basado en el patron de AVA ai voice for asterisk.

Audio: PCM16 16kHz base64 (input). PCM16 16kHz base64 (output).
"""

from __future__ import annotations

import asyncio
import audioop
import base64
import contextlib
import json
import time
from typing import TYPE_CHECKING, Any, Optional

from ..observability.log_setup import get_logger

logger = get_logger(__name__)

try:
    import aiohttp
except ImportError:
    aiohttp = None  # type: ignore[assignment]

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
    from ..config.models import ElevenLabsConvProviderCfg, AudioProfileCfg


def build_service(
    resolved: "ElevenLabsConvProviderCfg",
    audio_profile: "AudioProfileCfg",
    *,
    context_prompt: str = "",
    context_greeting: str = "",
) -> Any:
    """Factory for PipelineFactory. Returns an ElevenLabsConversationalService."""
    params = dict(resolved.params)
    agent_id = params.pop("agent_id", "")
    return ElevenLabsConversationalService(
        api_key=resolved.api_key,
        agent_id=agent_id,
        sample_rate=audio_profile.in_rate,
        output_sample_rate=audio_profile.out_rate,
        system_prompt=context_prompt or params.pop("system_prompt", ""),
        first_message=context_greeting or params.pop("first_message", ""),
        **params,
    )


class ElevenLabsConversationalService(FrameProcessor):
    """Full-agent: STT + LLM + TTS + VAD via ElevenLabs ConvAI WebSocket."""

    CONVAI_WS_URL = "wss://api.elevenlabs.io/v1/convai/conversation"

    def __init__(
        self,
        *,
        api_key: str,
        agent_id: str = "",
        sample_rate: int = 8000,
        output_sample_rate: int = 16000,
        system_prompt: str = "",
        first_message: str = "",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._api_key = api_key
        self._agent_id = agent_id
        self._sample_rate = sample_rate
        self._output_sample_rate = output_sample_rate
        self._system_prompt = system_prompt
        self._first_message = first_message

        self._ws: Any = None
        self._receive_task: Optional[asyncio.Task] = None
        self._closing = False
        self._audio_burst = False

        # Resample state: 8kHz input → 16kHz for ElevenLabs
        self._resample_in_state = None

        # Local energy VAD state
        self._vad_speech_count = 0
        self._vad_silence_count = 0
        self._vad_active = False
        self._vad_rms_threshold = 800
        self._vad_start_frames = 10
        self._vad_stop_frames = 30

        # OTel: agent_e2e_turn_around tracking
        self._agent_e2e_span: Any = None
        self._agent_e2e_start: float = 0.0

    async def _connect(self):
        if self._ws:
            return

        # Get signed URL for authenticated agent
        signed_url = await self._get_signed_url()

        try:
            self._ws = await ws_connect(
                signed_url,
                max_size=16 * 1024 * 1024,
                ping_interval=20,
                ping_timeout=20,
                close_timeout=5,
            )
            logger.info(f"{self}: Connected to ElevenLabs ConvAI")
            self._receive_task = self.create_task(self._receive_loop())

            # Send conversation initiation
            await self._send_init()
        except Exception as e:
            logger.error(f"{self}: Connection failed: {e}")

    async def _get_signed_url(self) -> str:
        if not self._agent_id:
            return self.CONVAI_WS_URL
        if aiohttp is None:
            return f"{self.CONVAI_WS_URL}?agent_id={self._agent_id}"
        url = f"https://api.elevenlabs.io/v1/convai/conversation/get_signed_url?agent_id={self._agent_id}"
        headers = {"xi-api-key": self._api_key}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        signed = data.get("signed_url", "")
                        if signed:
                            logger.info(f"{self}: Got signed URL")
                            return signed
        except Exception as e:
            logger.warning(f"{self}: Failed signed URL, using direct: {e}")
        return f"{self.CONVAI_WS_URL}?agent_id={self._agent_id}"

    async def _send_init(self):
        # Send minimal init — let ElevenLabs use dashboard config
        msg = {"type": "conversation_initiation_client_data", "dynamic_variables": {}}
        await self._ws.send(json.dumps(msg))
        logger.info(f"{self}: Sent conversation init (minimal)")

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
                if isinstance(message, str):
                    await self._handle_json(message)
        except ws_exceptions.ConnectionClosed as e:
            logger.info(f"{self}: WebSocket closed: code={e.code} reason={e.reason}")
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

        if t == "audio":
            audio_event = msg.get("audio_event", {})
            audio_b64 = audio_event.get("audio_base_64", "")
            if audio_b64:
                pcm16 = base64.b64decode(audio_b64)
                # ElevenLabs outputs 16kHz PCM
                frame = TTSAudioRawFrame(pcm16, self._output_sample_rate, 1)
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

        elif t == "interruption":
            await self.push_frame(InterruptionFrame())

        elif t == "user_transcript":
            transcript_event = msg.get("user_transcription_event", {})
            text = transcript_event.get("user_transcript", "") or msg.get("user_transcript", "")
            if text:
                await self.push_frame(TranscriptionFrame(text, "", time_now_iso8601()))
            # Start agent_e2e_turn_around OTel span
            self._agent_e2e_start = time.monotonic()
            try:
                from ..observability.otel import get_tracer
                self._agent_e2e_span = get_tracer().start_span("agent_e2e_turn_around")
            except Exception:
                self._agent_e2e_span = None

        elif t == "agent_response":
            response = msg.get("agent_response_event", {})
            text = response.get("agent_response", "")
            if text:
                logger.debug(f"{self}: Agent: {text[:80]}")

        elif t == "conversation_initiation_metadata":
            metadata = msg.get("conversation_initiation_metadata_event", {})
            logger.info(f"{self}: Conversation init: {metadata.get('conversation_id', '')}")

        elif t == "ping":
            ping_event = msg.get("ping_event", {})
            event_id = ping_event.get("event_id")
            if event_id and self._ws:
                await self._ws.send(json.dumps({"type": "pong", "event_id": event_id}))

        elif t in ("error", "Error"):
            logger.error(f"{self}: ElevenLabs error: {json.dumps(msg)}")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, InputAudioRawFrame):
            # Local energy VAD for fast interruption
            rms = audioop.rms(frame.audio, 2)
            if rms > self._vad_rms_threshold:
                self._vad_speech_count += 1
                self._vad_silence_count = 0
                if self._vad_speech_count >= self._vad_start_frames and not self._vad_active:
                    self._vad_active = True
                    await self.push_frame(InterruptionFrame())
                    # Also send interrupt to ElevenLabs
                    if self._ws and self._ws.state == WsState.OPEN:
                        await self._ws.send(json.dumps({"type": "interrupt"}))
            else:
                self._vad_silence_count += 1
                self._vad_speech_count = 0
                if self._vad_silence_count >= self._vad_stop_frames and self._vad_active:
                    self._vad_active = False

            # Send to ElevenLabs (PCM16 → resample 8k→16k → base64)
            if self._ws and self._ws.state == WsState.OPEN:
                pcm16 = frame.audio
                if self._sample_rate != 16000:
                    from ..audio.resampler import resample_audio
                    pcm16, self._resample_in_state = resample_audio(
                        pcm16, self._sample_rate, 16000, state=self._resample_in_state,
                    )
                audio_b64 = base64.b64encode(pcm16).decode("utf-8")
                await self._ws.send(json.dumps({"user_audio_chunk": audio_b64}))
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
