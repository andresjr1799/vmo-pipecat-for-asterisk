"""MetricsFrameProcessor — pass-through that tracks STT/LLM/TTS stage timing.

Se coloca en el pipeline despues de cada servicio (STT, LLM, TTS)
para trackear cuando se emiten los primeros frames de cada etapa.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from pipecat.frames.frames import (
    Frame, TranscriptionFrame, TextFrame, TTSAudioRawFrame,
    OutputAudioRawFrame, InterruptionFrame, ErrorFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

if TYPE_CHECKING:
    from ..call.metrics import CallMetrics


class MetricsFrameProcessor(FrameProcessor):
    """Pass-through processor that tracks pipeline stage latencies.

    Colocar la misma instancia despues de STT, LLM, y TTS.
    Detecta el tipo de frame para saber que etapa emitio.
    """

    def __init__(self, call_metrics: "CallMetrics", stage: str, **kwargs):
        super().__init__(**kwargs)
        self._cm = call_metrics
        self._stage = stage
        self._first = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # Track first frame arrival as STT start baseline
        if self._stage == "stt" and self._cm._turn_stt_start == 0.0:
            self._cm.stt_start()

        if isinstance(frame, InterruptionFrame):
            self._cm.record_barge_in()
            self._first = False
            self._cm.stt_start()  # reset for next turn

        elif isinstance(frame, ErrorFrame):
            self._cm.record_error()

        elif self._stage == "stt" and isinstance(frame, TranscriptionFrame):
            if not self._first and frame.text:
                self._first = True
                self._cm.stt_final(frame.text)

        elif self._stage == "llm" and isinstance(frame, TextFrame):
            if not self._first:
                self._first = True
                self._cm.llm_first_token()

        elif self._stage == "tts" and isinstance(frame, (TTSAudioRawFrame, OutputAudioRawFrame)):
            if not self._first:
                self._first = True
                self._cm.tts_first_audio()

        await self.push_frame(frame, direction)
