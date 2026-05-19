"""Provider wrapper: OpenAI LLM (PipeCat 1.1.0+).

1.1.0: usa settings=OpenAILLMService.Settings(model=...) y OPENAI_API_KEY del env.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..config.models import OpenAIProviderCfg, AudioProfileCfg

try:
    from pipecat.services.openai.llm import OpenAILLMService
    _PIPECAT = True
except ImportError:
    _PIPECAT = False

    class OpenAILLMService:  # type: ignore[no-redef]
        def __init__(self, **kw): self._kw = kw
        def register_function(self, name, cb): pass
        def link(self, n): pass


def build_service(resolved: "OpenAIProviderCfg", audio_profile: "AudioProfileCfg") -> Any:
    params = dict(resolved.params)
    model = params.pop("model", "gpt-4o-mini")
    temperature = params.pop("temperature", None)
    max_tokens = params.pop("max_tokens", None)

    # PipeCat 1.1.0 usa OPENAI_API_KEY del entorno (openai SDK estándar)
    if resolved.api_key:
        os.environ["OPENAI_API_KEY"] = resolved.api_key

    settings_kwargs: dict = {"model": model}
    if temperature is not None:
        settings_kwargs["temperature"] = float(temperature)
    if max_tokens is not None:
        settings_kwargs["max_tokens"] = int(max_tokens)

    return OpenAILLMService(
        settings=OpenAILLMService.Settings(**settings_kwargs)
    )
