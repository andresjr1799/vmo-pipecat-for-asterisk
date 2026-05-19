"""Provider wrapper: ElevenLabs TTS — WebSocket (PipeCat 1.1.0+).

Params soportados desde tenants.yaml:
  voice_id, model_id, speed, stability, similarity_boost,
  optimize_streaming_latency (solo HTTP, ignorado en WS)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..config.models import ElevenLabsProviderCfg, AudioProfileCfg

try:
    from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
    _PIPECAT = True
except ImportError:
    _PIPECAT = False

    class ElevenLabsTTSService:  # type: ignore[no-redef]
        def __init__(self, **kw): self._kw = kw
        def link(self, n): pass


def build_service(resolved: "ElevenLabsProviderCfg", audio_profile: "AudioProfileCfg") -> Any:
    params = dict(resolved.params)
    voice_id = params.pop("voice_id", "")
    model = params.pop("model_id", None) or params.pop("model", "eleven_multilingual_v2")
    speed = params.pop("speed", None)
    stability = params.pop("stability", None)
    similarity_boost = params.pop("similarity_boost", None)

    settings_kwargs: dict[str, Any] = {"voice": voice_id, "model": model}
    if speed is not None:
        settings_kwargs["speed"] = float(speed)
    if stability is not None:
        settings_kwargs["stability"] = float(stability)
    if similarity_boost is not None:
        settings_kwargs["similarity_boost"] = float(similarity_boost)

    return ElevenLabsTTSService(
        api_key=resolved.api_key,
        sample_rate=audio_profile.out_rate,
        settings=ElevenLabsTTSService.Settings(**settings_kwargs),
    )
