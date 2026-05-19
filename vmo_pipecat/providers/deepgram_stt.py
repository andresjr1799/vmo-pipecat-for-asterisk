"""Provider wrapper: Deepgram STT (PipeCat 1.1.0+).

El transport envia PCM16 (linear16) 8kHz directamente desde AudioSocket (/c(slin)).
Deepgram recibe linear16 8kHz nativamente sin conversion μ-law intermedia.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..config.models import DeepgramProviderCfg, AudioProfileCfg

try:
    from pipecat.services.deepgram.stt import DeepgramSTTService
    _PIPECAT = True
except ImportError:
    _PIPECAT = False

    class DeepgramSTTService:  # type: ignore[no-redef]
        def __init__(self, **kw): self._kw = kw
        def link(self, n): pass


def build_service(resolved: "DeepgramProviderCfg", audio_profile: "AudioProfileCfg") -> Any:
    """Deepgram STT con linear16 8kHz (PCM16 nativo de AudioSocket /c(slin)).

    El AsteriskAudioSocketInputTransport envia PCM16 8kHz sin conversion.
    Deepgram recibe encoding=linear16 sample_rate=8000.
    """
    params = dict(resolved.params)
    model = params.pop("model", None)
    language = params.pop("language", "es")

    settings_kwargs: dict[str, Any] = {"language": language}
    if model:
        settings_kwargs["model"] = model
    if "endpointing" in params:
        settings_kwargs["endpointing"] = params.pop("endpointing")
    else:
        settings_kwargs["endpointing"] = 600

    extra: dict[str, Any] = {}
    known = {"model", "language", "endpointing"}
    for key, value in params.items():
        if key not in known:
            extra[key] = value
    if extra:
        settings_kwargs["extra"] = extra

    return DeepgramSTTService(
        api_key=resolved.api_key,
        encoding="linear16",
        sample_rate=8000,
        settings=DeepgramSTTService.Settings(**settings_kwargs),
    )
