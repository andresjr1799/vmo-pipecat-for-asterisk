"""
AsteriskAudioSocketOutputTransport — PipeCat 1.1.0 output transport para AudioSocket.

Pacing a velocidad real-time (20ms por chunk de 320 bytes @ 8kHz) para
no saturar el buffer de Asterisk que solo almacena ~2s de audio.
"""

from __future__ import annotations

import asyncio
from typing import Optional, TYPE_CHECKING

try:
    from pipecat.transports.base_output import BaseOutputTransport
    from pipecat.transports.base_transport import TransportParams
    from pipecat.frames.frames import OutputAudioRawFrame
    _PIPECAT = True
except ImportError:
    _PIPECAT = False
    class TransportParams:
        def __init__(self, **kw): pass
    class BaseOutputTransport:
        def __init__(self, params=None, **kw): self._params = params
        async def write_audio_frame(self, frame) -> bool: return False
        async def start(self, frame=None): pass
        async def set_transport_ready(self, frame=None): pass
    class OutputAudioRawFrame:
        def __init__(self, audio, sample_rate, num_channels, **kw): self.audio = audio

if TYPE_CHECKING:
    from ..audio.audiosocket_server import AudioSocketServer

_CHUNK_MS = 20      # ms per chunk
_CHUNK_BYTES = 320  # 20ms @ 8kHz PCM16 = 320 bytes
_PACE_SEC = 0.018  # 18ms sleep + ~2ms TCP write = ~20ms por chunk


class AsteriskAudioSocketOutputTransport(BaseOutputTransport):

    def __init__(self, audiosocket, params, *, out_sample_rate=8000, channels=1, **kwargs):
        super().__init__(params, **kwargs)
        self._audiosocket = audiosocket
        self._out_sample_rate = out_sample_rate
        self._channels = channels
        self._conn_id: Optional[str] = None

    def bind(self, conn_id): self._conn_id = conn_id

    @property
    def conn_id(self): return self._conn_id

    async def start(self, frame=None):
        await super().start(frame)
        await self.set_transport_ready(frame)

    async def stop(self, frame=None):
        self._conn_id = None
        await super().stop(frame)

    async def cancel(self, frame=None):
        self._conn_id = None
        await super().cancel(frame)

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
        audio = getattr(frame, "audio", None)
        if not self._conn_id or not audio:
            return False

        import struct

        if self._out_sample_rate == 8000:
            pcm8k = audio
        else:
            fmt_in = f"<{len(audio)//2}h"
            samples = struct.unpack(fmt_in, audio)
            pcm8k = struct.pack(f"<{len(samples[::2])}h", *samples[::2])

        for i in range(0, len(pcm8k), _CHUNK_BYTES):
            chunk = pcm8k[i:i + _CHUNK_BYTES]
            await self._audiosocket.send_audio(self._conn_id, chunk)
            await asyncio.sleep(_PACE_SEC)

        return True
