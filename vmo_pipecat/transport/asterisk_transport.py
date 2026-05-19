"""
AsteriskAudioSocketTransport — factory que combina input y output para PipeCat 1.x.

Uso en PipelineFactory:
    transport = AsteriskAudioSocketTransport(audiosocket, audio_profile)
    pipeline  = Pipeline([
        transport.input(),   # AsteriskAudioSocketInputTransport
        agent_or_vad_stt_llm_tts,
        transport.output(),  # AsteriskAudioSocketOutputTransport
    ])
    transport.bind(conn_id)  # tras el handshake UUID
"""

from __future__ import annotations

from typing import TYPE_CHECKING

try:
    from pipecat.transports.base_transport import TransportParams
except ImportError:
    class TransportParams:  # type: ignore[no-redef]
        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)

from .input import AsteriskAudioSocketInputTransport
from .output import AsteriskAudioSocketOutputTransport

if TYPE_CHECKING:
    from ..audio.audiosocket_server import AudioSocketServer
    from ..config.models import AudioProfileCfg


class AsteriskAudioSocketTransport:
    """Combina input + output transport para una llamada AudioSocket.

    Compatible con PipeCat 1.1.0+ (TransportParams en base_transport,
    BaseInputTransport en base_input, BaseOutputTransport en base_output).
    """

    def __init__(
        self,
        audiosocket: "AudioSocketServer",
        audio_profile: "AudioProfileCfg",
    ) -> None:
        params = TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=audio_profile.in_rate,
            audio_out_sample_rate=audio_profile.out_rate,
            audio_in_channels=audio_profile.channels,
            audio_out_channels=audio_profile.channels,
            # CRÍTICO: sin passthrough=True, el audio va a _audio_in_queue
            # pero el task solo lo reenvía si audio_in_passthrough=True.
            # Con False (default), el audio nunca llega a DeepgramSTTService.
            audio_in_passthrough=True,
        )
        self._input = AsteriskAudioSocketInputTransport(
            params,
            in_sample_rate=audio_profile.in_rate,
            channels=audio_profile.channels,
        )
        self._output = AsteriskAudioSocketOutputTransport(
            audiosocket,
            params,
            out_sample_rate=audio_profile.out_rate,
            channels=audio_profile.channels,
        )

    def input(self) -> AsteriskAudioSocketInputTransport:
        return self._input

    def output(self) -> AsteriskAudioSocketOutputTransport:
        return self._output

    def bind(self, conn_id: str) -> None:
        self._input.bind(conn_id)
        self._output.bind(conn_id)

    async def push_audio(self, audio_bytes: bytes) -> None:
        await self._input.push_audio(audio_bytes)
