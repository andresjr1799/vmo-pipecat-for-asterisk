"""
Pydantic v2 configuration models for tenants.yaml.

Schema follows §6 of the architecture document.
Cross-validations enforce referential integrity at load time.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Optional, Union
from pydantic import BaseModel, Field, model_validator


# ── Audio ─────────────────────────────────────────────────────────────────────

class AudioProfileCfg(BaseModel):
    in_rate: int = 8000
    out_rate: int = 8000
    codec: str = "slin"
    channels: int = 1


class AudioSocketCfg(BaseModel):
    bind_host: str = "0.0.0.0"
    port: int = 8090
    advertise_host: str = "vmo_pipecat"


# ── Asterisk nodes ─────────────────────────────────────────────────────────────

class AsteriskNodeCfg(BaseModel):
    id: str
    host: str
    ari_port: int = 8088
    ari_username: str
    ari_password: str
    ari_app: str = "vmo-pipecat-app"
    ari_scheme: str = "http"


# ── Defaults ──────────────────────────────────────────────────────────────────

class VADDefaultsCfg(BaseModel):
    silence_threshold_ms: int = 200    # stop_secs en ms
    speech_threshold_ms: int = 150     # start_secs en ms


class InterruptionDefaultsCfg(BaseModel):
    strategy: str = "min_words"   # never | always | min_words
    min_words: int = 3


class DefaultsCfg(BaseModel):
    vad: VADDefaultsCfg = Field(default_factory=VADDefaultsCfg)
    interruption: InterruptionDefaultsCfg = Field(default_factory=InterruptionDefaultsCfg)
    audio_profile: str = "telephony_8k"
    stasis_app: str = "vmo-pipecat-app"


# ── Provider configs ───────────────────────────────────────────────────────────
# Each entry = one (service, credential) pair. N entries of same kind are valid.

class DeepgramProviderCfg(BaseModel):
    kind: Literal["deepgram"]
    mode: Literal["stt", "tts"]
    api_key: str = ""
    params: dict[str, Any] = Field(default_factory=dict)


class AWSTranscribeProviderCfg(BaseModel):
    kind: Literal["aws_transcribe"]
    mode: Literal["stt"]
    region: str = "us-east-1"
    access_key_id: str = ""
    secret_access_key: str = ""
    params: dict[str, Any] = Field(default_factory=dict)


class ElevenLabsProviderCfg(BaseModel):
    kind: Literal["elevenlabs"]
    mode: Literal["tts"]
    api_key: str = ""
    params: dict[str, Any] = Field(default_factory=dict)


class ElevenLabsSTTProviderCfg(BaseModel):
    kind: Literal["elevenlabs_stt"]
    mode: Literal["stt"]
    api_key: str = ""
    params: dict[str, Any] = Field(default_factory=dict)


class OpenAIProviderCfg(BaseModel):
    kind: Literal["openai"]
    mode: Literal["llm"]
    api_key: str = ""
    params: dict[str, Any] = Field(default_factory=dict)


class _AgenticBusAuthCfg(BaseModel):
    type: str = "bearer"   # bearer | basic | aws_sigv4 | mtls | none
    token: str = ""


class _AgenticBusConnectionCfg(BaseModel):
    servers: str = ""
    jetstream: bool = False


class _AgenticBusOutboundCfg(BaseModel):
    destination: str


class _AgenticBusInboundCfg(BaseModel):
    destination: str
    group: str = "vmo-pipecat"


class AgenticBusProviderCfg(BaseModel):
    kind: Literal["agentic_bus"]
    mode: Literal["llm"]
    transport: str  # nats | rabbitmq | kinesis | kafka | sqs
    outbound: _AgenticBusOutboundCfg
    inbound: _AgenticBusInboundCfg
    auth: _AgenticBusAuthCfg = Field(default_factory=_AgenticBusAuthCfg)
    connection: _AgenticBusConnectionCfg = Field(default_factory=_AgenticBusConnectionCfg)
    params: dict[str, Any] = Field(default_factory=dict)


class DeepgramVoiceAgentProviderCfg(BaseModel):
    kind: Literal["deepgram_voice_agent"]
    mode: Literal["full_agent"]
    api_key: str = ""
    params: dict[str, Any] = Field(default_factory=dict)


class ElevenLabsConvProviderCfg(BaseModel):
    kind: Literal["elevenlabs_conv"]
    mode: Literal["full_agent"]
    api_key: str = ""
    params: dict[str, Any] = Field(default_factory=dict)


ProviderCfg = Annotated[
    Union[
        DeepgramProviderCfg,
        AWSTranscribeProviderCfg,
        ElevenLabsProviderCfg,
        ElevenLabsSTTProviderCfg,
        OpenAIProviderCfg,
        AgenticBusProviderCfg,
        DeepgramVoiceAgentProviderCfg,
        ElevenLabsConvProviderCfg,
    ],
    Field(discriminator="kind"),
]


# ── Pipeline overrides (per-pipeline, optional) ────────────────────────────────

class VADCfg(BaseModel):
    stop_secs: Optional[float] = None     # override defaults.vad.silence_threshold_ms
    start_secs: Optional[float] = None    # override defaults.vad.speech_threshold_ms


class InterruptionCfg(BaseModel):
    strategy: str = "min_words"
    min_words: int = 3


# ── Pipeline configs ───────────────────────────────────────────────────────────

class ModularPipelineCfg(BaseModel):
    kind: Literal["modular"]
    stt: str                                          # provider id
    llm: str                                          # provider id
    tts: str                                          # provider id
    vad: str = "silero"                               # silero | asterisk_talk_detect | none
    vad_params: Optional[VADCfg] = None               # override de stop_secs/start_secs
    interruption: Optional[InterruptionCfg] = None


class FullAgentPipelineCfg(BaseModel):
    kind: Literal["full_agent"]
    provider: str                                     # provider id


PipelineCfg = Annotated[
    Union[ModularPipelineCfg, FullAgentPipelineCfg],
    Field(discriminator="kind"),
]


# ── Context ────────────────────────────────────────────────────────────────────

class ContextCfg(BaseModel):
    prompt: str = ""
    greeting: str = ""
    audio_profile: str = "telephony_8k"
    tools: list[str] = Field(default_factory=list)


# ── Tenant ─────────────────────────────────────────────────────────────────────

class RouteCfg(BaseModel):
    pipeline: str
    context: str


class TransferCfg(BaseModel):
    enabled: bool = False
    context: str = "from-vmo-transfer"
    default_target: str = "9000"


class VADOverridesCfg(BaseModel):
    silence_threshold_ms: Optional[int] = None
    speech_threshold_ms: Optional[int] = None


class InterruptionOverridesCfg(BaseModel):
    strategy: Optional[str] = None
    min_words: Optional[int] = None


class OverridesCfg(BaseModel):
    vad: Optional[VADOverridesCfg] = None
    interruption: Optional[InterruptionOverridesCfg] = None


class TenantCfg(BaseModel):
    name: str
    transfer: TransferCfg = Field(default_factory=TransferCfg)
    routes: dict[str, RouteCfg]
    overrides: OverridesCfg = Field(default_factory=OverridesCfg)


# ── Fallback ───────────────────────────────────────────────────────────────────

class FallbackCfg(BaseModel):
    policy: Literal["reject", "use_tenant"] = "reject"
    tenant: Optional[str] = None
    reject_message: str = "Su llamada no puede ser atendida en este momento."


# ── Root config ────────────────────────────────────────────────────────────────

class TenantsConfig(BaseModel):
    defaults: DefaultsCfg = Field(default_factory=DefaultsCfg)
    asterisk_nodes: list[AsteriskNodeCfg] = Field(default_factory=list)
    audiosocket: AudioSocketCfg = Field(default_factory=AudioSocketCfg)
    audio_profiles: dict[str, AudioProfileCfg] = Field(default_factory=dict)
    providers: dict[str, ProviderCfg] = Field(default_factory=dict)
    pipelines: dict[str, PipelineCfg] = Field(default_factory=dict)
    contexts: dict[str, ContextCfg] = Field(default_factory=dict)
    tenants: dict[str, TenantCfg] = Field(default_factory=dict)
    fallback: Optional[FallbackCfg] = None

    @model_validator(mode="after")
    def _validate_cross_references(self) -> "TenantsConfig":
        errors: list[str] = []

        # Pipelines → providers
        for pid, pipeline in self.pipelines.items():
            if isinstance(pipeline, ModularPipelineCfg):
                for slot, expected_mode in [("stt", "stt"), ("llm", "llm"), ("tts", "tts")]:
                    ref = getattr(pipeline, slot)
                    if ref not in self.providers:
                        errors.append(f"pipeline '{pid}': {slot}='{ref}' not in providers")
                    elif self.providers[ref].mode != expected_mode:
                        actual = self.providers[ref].mode
                        errors.append(
                            f"pipeline '{pid}': provider '{ref}' has mode='{actual}', expected '{expected_mode}'"
                        )
            elif isinstance(pipeline, FullAgentPipelineCfg):
                ref = pipeline.provider
                if ref not in self.providers:
                    errors.append(f"pipeline '{pid}': provider='{ref}' not in providers")
                elif self.providers[ref].mode != "full_agent":
                    actual = self.providers[ref].mode
                    errors.append(
                        f"pipeline '{pid}': provider '{ref}' has mode='{actual}', expected 'full_agent'"
                    )

        # Tenants → pipelines / contexts; must have "default" route
        for tid, tenant in self.tenants.items():
            if "default" not in tenant.routes:
                errors.append(f"tenant '{tid}': missing 'default' route")
            for did, route in tenant.routes.items():
                if route.pipeline and route.pipeline not in self.pipelines:
                    errors.append(
                        f"tenant '{tid}', route '{did}': pipeline='{route.pipeline}' not in pipelines"
                    )
                if route.context and route.context not in self.contexts:
                    errors.append(
                        f"tenant '{tid}', route '{did}': context='{route.context}' not in contexts"
                    )

        # Contexts → audio_profiles
        for cid, ctx in self.contexts.items():
            if ctx.audio_profile and self.audio_profiles and ctx.audio_profile not in self.audio_profiles:
                errors.append(
                    f"context '{cid}': audio_profile='{ctx.audio_profile}' not in audio_profiles"
                )

        # Fallback → tenants
        if self.fallback and self.fallback.policy == "use_tenant":
            if not self.fallback.tenant:
                errors.append("fallback: policy='use_tenant' requires 'tenant' field")
            elif self.fallback.tenant not in self.tenants:
                errors.append(
                    f"fallback: tenant='{self.fallback.tenant}' not in tenants"
                )

        if errors:
            raise ValueError("Configuration cross-reference errors:\n" + "\n".join(f"  • {e}" for e in errors))

        return self
