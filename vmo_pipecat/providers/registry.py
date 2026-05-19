"""
Provider registry — maps kind → build_service function.

PipelineFactory calls build_provider(resolved_cfg, audio_profile) to get a
ready-to-use PipeCat service instance. Adding a new provider: implement
build_service() + register here. No other changes needed.
"""

from __future__ import annotations

from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from ..config.models import AudioProfileCfg

# ── Builder imports ────────────────────────────────────────────────────────────
from .deepgram_stt import build_service as _build_deepgram_stt
from .deepgram_tts import build_service as _build_deepgram_tts
from .deepgram_fullagent import build_service as _build_deepgram_fullagent
from .aws_transcribe_stt import build_service as _build_aws_transcribe
from .elevenlabs_tts import build_service as _build_elevenlabs_tts
from .elevenlabs_stt import build_service as _build_elevenlabs_stt
from .elevenlabs_fullagent import build_service as _build_elevenlabs_fullagent
from .openai_llm import build_service as _build_openai
from .agentic_bus_llm import build_service as _build_agentic_bus


def _build_deepgram(resolved: Any, audio_profile: "AudioProfileCfg") -> Any:
    """Dispatch deepgram kind by mode (stt | tts)."""
    if resolved.mode == "tts":
        return _build_deepgram_tts(resolved, audio_profile)
    return _build_deepgram_stt(resolved, audio_profile)


# ── Registry ───────────────────────────────────────────────────────────────────

PROVIDER_BUILDERS: dict[str, Callable[..., Any]] = {
    "deepgram":             _build_deepgram,
    "aws_transcribe":       _build_aws_transcribe,
    "elevenlabs":           _build_elevenlabs_tts,
    "elevenlabs_stt":       _build_elevenlabs_stt,
    "openai":               _build_openai,
    "agentic_bus":          _build_agentic_bus,
    "deepgram_voice_agent": _build_deepgram_fullagent,
    "elevenlabs_conv":      _build_elevenlabs_fullagent,
}


def build_provider(resolved: Any, audio_profile: "AudioProfileCfg", **kwargs: Any) -> Any:
    """Instantiate the correct PipeCat service for a resolved provider config.

    Extra kwargs (e.g. context_prompt, context_greeting for full-agent builders)
    are forwarded to the builder if it accepts them.
    """
    builder = PROVIDER_BUILDERS.get(resolved.kind)
    if builder is None:
        raise ValueError(f"Unknown provider kind: '{resolved.kind}'")
    if kwargs:
        try:
            return builder(resolved, audio_profile, **kwargs)
        except TypeError:
            return builder(resolved, audio_profile)
    return builder(resolved, audio_profile)


def register(kind: str, builder: Callable[..., Any]) -> None:
    """Register a custom provider builder (for extensions/testing)."""
    PROVIDER_BUILDERS[kind] = builder
