"""Provider wrapper: AWS Transcribe STT.

AWSTranscribeSTTService acepta:
  - api_key: AWS secret access key (o via env AWS_SECRET_ACCESS_KEY)
  - aws_access_key_id: AWS access key ID (o via env AWS_ACCESS_KEY_ID)
  - aws_session_token: (opcional, para credenciales temporales)
  - region: region AWS (o via env AWS_DEFAULT_REGION)
  - sample_rate: 8000 o 16000 (AWS solo soporta estos dos)

Si no se pasan credenciales explicitas, usa la cadena default de boto3
(env vars, instance profiles, IRSA, etc.).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..config.models import AWSTranscribeProviderCfg, AudioProfileCfg

try:
    from pipecat.services.aws.stt import AWSTranscribeSTTService
    _PIPECAT = True
except ImportError:
    _PIPECAT = False

    class AWSTranscribeSTTService:  # type: ignore[no-redef]
        def __init__(self, **kw): self._kw = kw
        def link(self, n): pass


def build_service(resolved: "AWSTranscribeProviderCfg", audio_profile: "AudioProfileCfg") -> Any:
    params = dict(resolved.params)
    language = params.pop("language", "es-US")

    kwargs: dict[str, Any] = {
        "region": resolved.region,
        "language": language,
        "sample_rate": audio_profile.in_rate,
    }

    # Credenciales explicitas (opcional — boto3 las busca en env si no se pasan)
    if resolved.access_key_id:
        kwargs["aws_access_key_id"] = resolved.access_key_id
    if resolved.secret_access_key:
        kwargs["api_key"] = resolved.secret_access_key  # Pipecat: api_key = secret key

    return AWSTranscribeSTTService(**kwargs, **params)
