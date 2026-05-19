"""Provider wrapper: ElevenLabs STT (PipeCat 1.1.0+)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..config.models import ElevenLabsSTTProviderCfg, AudioProfileCfg

try:
    import aiohttp
    from pipecat.services.elevenlabs.stt import ElevenLabsSTTService
    _PIPECAT = True
except ImportError:
    _PIPECAT = False
    aiohttp = None  # type: ignore[assignment]

    class ElevenLabsSTTService:  # type: ignore[no-redef]
        def __init__(self, **kw): self._kw = kw
        def link(self, n): pass


def build_service(resolved: Any, audio_profile: "AudioProfileCfg") -> Any:
    # ElevenLabsSTTService requiere aiohttp_session — crear una por llamada
    session = aiohttp.ClientSession() if aiohttp else None
    return ElevenLabsSTTService(
        api_key=resolved.api_key,
        aiohttp_session=session,
        sample_rate=audio_profile.in_rate,
        settings=ElevenLabsSTTService.Settings(language="es"),
    )
