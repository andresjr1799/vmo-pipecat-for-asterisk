"""
TenantResolver — resolves (tenant_id, DID) → SessionConfig (frozen).

SessionConfig is captured ONCE at StasisStart and never re-read from ConfigStore
during the active call. This is the hot-reload isolation guarantee (§7.2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, TYPE_CHECKING

from ..config.models import (
    AudioProfileCfg,
    ContextCfg,
    FullAgentPipelineCfg,
    ModularPipelineCfg,
    OverridesCfg,
    TenantsConfig,
    TransferCfg,
)
from ..config.store import ConfigStore

if TYPE_CHECKING:
    from ..call.identity import CallIdentity


# ── SessionConfig ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SessionConfig:
    """Frozen snapshot of everything a call needs for its lifetime.

    Built once at StasisStart; no field is ever re-read from ConfigStore.
    `identity` is None during Phase 1 and populated in Phase 3 (CallLifecycle).
    """
    context: ContextCfg
    pipeline: Any                         # ModularPipelineCfg | FullAgentPipelineCfg
    providers: Mapping[str, Any]          # provider id → resolved ProviderCfg
    audio_profile: AudioProfileCfg
    transfer: TransferCfg
    overrides: OverridesCfg
    config_version: int
    identity: Optional["CallIdentity"] = field(default=None, compare=False)


# ── TenantResolver ─────────────────────────────────────────────────────────────

class TenantResolverError(Exception):
    """Raised when a call cannot be routed (and fallback=reject)."""


class TenantResolver:
    """Resolves (tenant_id, did) → SessionConfig from the ConfigStore snapshot."""

    def __init__(self, store: ConfigStore) -> None:
        self._store = store

    def resolve(
        self,
        tenant_id: str,
        did: str,
        identity: Optional["CallIdentity"] = None,
    ) -> SessionConfig:
        """Resolve configuration for a call.

        Reads ConfigStore.current ONCE and freezes everything into SessionConfig.
        After this returns, the call is isolated from future hot-reloads.

        Raises:
            TenantResolverError: when fallback.policy='reject' and tenant unknown.
            ValueError: when required config sections are missing.
        """
        cfg = self._store.current
        if cfg is None:
            raise TenantResolverError("ConfigStore has no loaded configuration")

        # 1. Resolve tenant (with fallback)
        tenant = cfg.tenants.get(tenant_id)
        if tenant is None:
            tenant, tenant_id = self._apply_fallback(cfg, tenant_id)

        # 2. Resolve route (DID → default)
        route = tenant.routes.get(did) or tenant.routes.get("default")
        if route is None:
            raise TenantResolverError(
                f"tenant='{tenant_id}' has no route for did='{did}' and no 'default'"
            )

        # 3. Pipeline, context, audio_profile
        pipeline = cfg.pipelines[route.pipeline]
        context = cfg.contexts[route.context]

        profile_name = context.audio_profile or cfg.defaults.audio_profile
        audio_profile = cfg.audio_profiles.get(profile_name) or AudioProfileCfg()

        # 4. Resolve providers referenced by pipeline
        providers: dict[str, Any] = {}
        if isinstance(pipeline, ModularPipelineCfg):
            for slot in ("stt", "llm", "tts"):
                pid = getattr(pipeline, slot)
                providers[pid] = cfg.providers[pid]
        elif isinstance(pipeline, FullAgentPipelineCfg):
            pid = pipeline.provider
            providers[pid] = cfg.providers[pid]

        # 5. Overrides: defaults ← tenant overrides
        overrides = self._merge_overrides(cfg, tenant.overrides)

        return SessionConfig(
            context=context,
            pipeline=pipeline,
            providers=providers,
            audio_profile=audio_profile,
            transfer=tenant.transfer,
            overrides=overrides,
            config_version=self._store.version,
            identity=identity,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_fallback(cfg: TenantsConfig, original_tenant_id: str):
        fb = cfg.fallback
        if fb is None or fb.policy == "reject":
            raise TenantResolverError(
                f"Unknown tenant='{original_tenant_id}' and fallback.policy='reject'"
            )
        # policy == "use_tenant"
        fallback_tenant = cfg.tenants.get(fb.tenant or "")
        if fallback_tenant is None:
            raise TenantResolverError(
                f"Unknown tenant='{original_tenant_id}'; fallback.tenant='{fb.tenant}' also not found"
            )
        return fallback_tenant, fb.tenant

    @staticmethod
    def _merge_overrides(cfg: TenantsConfig, tenant_overrides: OverridesCfg) -> OverridesCfg:
        """Produces an OverridesCfg that contains the effective vad/interruption settings."""
        from ..config.models import VADOverridesCfg, InterruptionOverridesCfg

        d = cfg.defaults
        tv = tenant_overrides.vad
        ti = tenant_overrides.interruption

        vad = VADOverridesCfg(
            silence_threshold_ms=tv.silence_threshold_ms if (tv and tv.silence_threshold_ms is not None)
                                  else d.vad.silence_threshold_ms,
            speech_threshold_ms=tv.speech_threshold_ms if (tv and tv.speech_threshold_ms is not None)
                                 else d.vad.speech_threshold_ms,
        )
        interruption = InterruptionOverridesCfg(
            strategy=ti.strategy if (ti and ti.strategy is not None) else d.interruption.strategy,
            min_words=ti.min_words if (ti and ti.min_words is not None) else d.interruption.min_words,
        )
        return OverridesCfg(vad=vad, interruption=interruption)
