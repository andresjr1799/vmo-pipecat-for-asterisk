"""Unit tests for TenantResolver and SessionConfig."""

import pytest

from vmo_pipecat.config.models import TenantsConfig
from vmo_pipecat.config.store import ConfigStore
from vmo_pipecat.tenancy.resolver import TenantResolver, TenantResolverError, SessionConfig


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _make_store(extra: dict | None = None) -> ConfigStore:
    base = {
        "audio_profiles": {
            "telephony_8k": {"in_rate": 8000, "out_rate": 8000, "codec": "slin", "channels": 1}
        },
        "providers": {
            "stt_p": {"kind": "deepgram", "mode": "stt", "api_key": "dk1"},
            "llm_p": {"kind": "openai", "mode": "llm", "api_key": "ok1"},
            "tts_p": {"kind": "elevenlabs", "mode": "tts", "api_key": "ek1"},
            "stt_p2": {"kind": "deepgram", "mode": "stt", "api_key": "dk2"},
        },
        "pipelines": {
            "pipe_es": {
                "kind": "modular",
                "stt": "stt_p",
                "llm": "llm_p",
                "tts": "tts_p",
            },
            "pipe_en": {
                "kind": "modular",
                "stt": "stt_p2",
                "llm": "llm_p",
                "tts": "tts_p",
            },
        },
        "contexts": {
            "ctx_es": {"prompt": "Hola", "audio_profile": "telephony_8k"},
            "ctx_en": {"prompt": "Hello", "audio_profile": "telephony_8k"},
            "ctx_default": {"prompt": "Hi", "audio_profile": "telephony_8k"},
        },
        "tenants": {
            "acme": {
                "name": "Acme S.A.",
                "transfer": {"enabled": True, "context": "from-vmo-transfer", "default_target": "9000"},
                "routes": {
                    "1000": {"pipeline": "pipe_es", "context": "ctx_es"},
                    "2000": {"pipeline": "pipe_en", "context": "ctx_en"},
                    "default": {"pipeline": "pipe_es", "context": "ctx_default"},
                },
                "overrides": {"vad": {"silence_threshold_ms": 700}},
            },
            "globex": {
                "name": "Globex Corp",
                "routes": {
                    "default": {"pipeline": "pipe_en", "context": "ctx_default"},
                },
            },
        },
        "fallback": {"policy": "use_tenant", "tenant": "globex"},
    }
    if extra:
        base.update(extra)
    cfg = TenantsConfig.model_validate(base)
    store = ConfigStore()
    store.swap(cfg, source_path="test", sha256="abc")
    return store


@pytest.fixture
def store() -> ConfigStore:
    return _make_store()


@pytest.fixture
def resolver(store) -> TenantResolver:
    return TenantResolver(store)


# ── Resolve: happy path ────────────────────────────────────────────────────────

def test_resolve_known_tenant_and_did(resolver):
    sc = resolver.resolve("acme", "1000")
    assert isinstance(sc, SessionConfig)
    assert sc.pipeline.stt == "stt_p"
    assert sc.context.prompt == "Hola"
    assert sc.audio_profile.in_rate == 8000


def test_resolve_known_tenant_default_route(resolver):
    sc = resolver.resolve("acme", "9999")  # unmapped DID → default
    assert sc.context.prompt == "Hi"


def test_resolve_providers_included(resolver):
    sc = resolver.resolve("acme", "1000")
    assert "stt_p" in sc.providers
    assert "llm_p" in sc.providers
    assert "tts_p" in sc.providers
    assert sc.providers["stt_p"].api_key == "dk1"


def test_resolve_transfer_config(resolver):
    sc = resolver.resolve("acme", "1000")
    assert sc.transfer.enabled is True
    assert sc.transfer.default_target == "9000"


def test_resolve_config_version(resolver, store):
    sc = resolver.resolve("acme", "1000")
    assert sc.config_version == store.version


# ── Overrides merge ────────────────────────────────────────────────────────────

def test_tenant_vad_override_applied(resolver):
    sc = resolver.resolve("acme", "1000")
    # acme overrides silence_threshold_ms = 700 (default is 600)
    assert sc.overrides.vad.silence_threshold_ms == 700


def test_tenant_without_overrides_uses_defaults(resolver):
    sc = resolver.resolve("globex", "default")
    # globex has no overrides → defaults (600)
    assert sc.overrides.vad.silence_threshold_ms == 600


# ── Fallback ────────────────────────────────────────────────────────────────────

def test_unknown_tenant_with_use_tenant_fallback(resolver):
    sc = resolver.resolve("unknown_corp", "1000")
    # Falls back to globex
    assert sc.pipeline.stt == "stt_p2"


def test_unknown_tenant_with_reject_fallback():
    cfg = TenantsConfig.model_validate({
        "audio_profiles": {
            "telephony_8k": {"in_rate": 8000, "out_rate": 8000, "codec": "slin", "channels": 1}
        },
        "providers": {
            "stt_p": {"kind": "deepgram", "mode": "stt", "api_key": "x"},
            "llm_p": {"kind": "openai", "mode": "llm", "api_key": "x"},
            "tts_p": {"kind": "elevenlabs", "mode": "tts", "api_key": "x"},
        },
        "pipelines": {
            "pipe": {"kind": "modular", "stt": "stt_p", "llm": "llm_p", "tts": "tts_p"}
        },
        "contexts": {"ctx": {"prompt": "Hi", "audio_profile": "telephony_8k"}},
        "tenants": {
            "acme": {"name": "A", "routes": {"default": {"pipeline": "pipe", "context": "ctx"}}}
        },
        "fallback": {"policy": "reject"},
    })
    store = ConfigStore()
    store.swap(cfg, source_path="test", sha256="def")
    resolver = TenantResolver(store)
    with pytest.raises(TenantResolverError, match="fallback.policy='reject'"):
        resolver.resolve("unknown_corp", "1000")


# ── Multi-key same kind ────────────────────────────────────────────────────────

def test_multi_key_providers_distinct(resolver):
    # stt_p and stt_p2 are both deepgram but different api_keys
    sc1 = resolver.resolve("acme", "1000")
    sc2 = resolver.resolve("acme", "2000")
    assert sc1.providers["stt_p"].api_key == "dk1"
    assert sc2.providers["stt_p2"].api_key == "dk2"


# ── Edge cases ─────────────────────────────────────────────────────────────────

def test_resolve_without_loaded_config_fails():
    empty_store = ConfigStore()
    resolver = TenantResolver(empty_store)
    with pytest.raises(TenantResolverError, match="no loaded configuration"):
        resolver.resolve("acme", "1000")


def test_identity_passed_through(resolver):
    from vmo_pipecat.call.identity import CallIdentity
    identity = CallIdentity(
        vmo_call_id="uuid-1",
        asterisk_channel_id="ch-1",
        call_id_sbc="sbc-1",
        tenant_id="acme",
        tenant_name="Acme S.A.",
        node_id="ast-1",
        did="1000",
    )
    sc = resolver.resolve("acme", "1000", identity=identity)
    assert sc.identity is identity
