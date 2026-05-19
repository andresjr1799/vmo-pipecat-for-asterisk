"""Unit tests for vmo_pipecat.config.models (Pydantic v2 schema)."""

import pytest
from pydantic import ValidationError

from vmo_pipecat.config.models import (
    TenantsConfig,
    ModularPipelineCfg,
    FullAgentPipelineCfg,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _minimal() -> dict:
    """Minimal valid config that passes cross-reference validation."""
    return {
        "audio_profiles": {
            "telephony_8k": {"in_rate": 8000, "out_rate": 8000, "codec": "slin", "channels": 1}
        },
        "providers": {
            "stt_p": {"kind": "deepgram", "mode": "stt", "api_key": "x"},
            "llm_p": {"kind": "openai", "mode": "llm", "api_key": "x"},
            "tts_p": {"kind": "elevenlabs", "mode": "tts", "api_key": "x"},
        },
        "pipelines": {
            "my_pipeline": {
                "kind": "modular",
                "stt": "stt_p",
                "llm": "llm_p",
                "tts": "tts_p",
            }
        },
        "contexts": {
            "ctx": {"prompt": "Hello", "audio_profile": "telephony_8k"}
        },
        "tenants": {
            "acme": {
                "name": "Acme",
                "routes": {"default": {"pipeline": "my_pipeline", "context": "ctx"}},
            }
        },
    }


# ── Valid configs ──────────────────────────────────────────────────────────────

def test_minimal_config_loads():
    cfg = TenantsConfig.model_validate(_minimal())
    assert "acme" in cfg.tenants
    assert isinstance(cfg.pipelines["my_pipeline"], ModularPipelineCfg)


def test_multi_key_same_provider_kind():
    """N entries of the same kind (e.g. two deepgram STT) are valid."""
    d = _minimal()
    d["providers"]["stt_p2"] = {"kind": "deepgram", "mode": "stt", "api_key": "y"}
    cfg = TenantsConfig.model_validate(d)
    assert "stt_p2" in cfg.providers


def test_full_agent_pipeline():
    d = _minimal()
    d["providers"]["va"] = {"kind": "deepgram_voice_agent", "mode": "full_agent", "api_key": "x"}
    d["pipelines"]["fa_pipe"] = {"kind": "full_agent", "provider": "va"}
    d["tenants"]["acme"]["routes"]["1001"] = {"pipeline": "fa_pipe", "context": "ctx"}
    cfg = TenantsConfig.model_validate(d)
    assert isinstance(cfg.pipelines["fa_pipe"], FullAgentPipelineCfg)


def test_agentic_bus_provider():
    d = _minimal()
    d["providers"]["bus_p"] = {
        "kind": "agentic_bus",
        "mode": "llm",
        "transport": "nats",
        "outbound": {"destination": "agentic.requests.acme"},
        "inbound": {"destination": "agentic.responses.acme"},
        "connection": {"servers": "nats://localhost:4222"},
    }
    cfg = TenantsConfig.model_validate(d)
    assert cfg.providers["bus_p"].kind == "agentic_bus"


def test_fallback_use_tenant():
    d = _minimal()
    d["tenants"]["globex"] = {
        "name": "Globex",
        "routes": {"default": {"pipeline": "my_pipeline", "context": "ctx"}},
    }
    d["fallback"] = {"policy": "use_tenant", "tenant": "globex"}
    cfg = TenantsConfig.model_validate(d)
    assert cfg.fallback.policy == "use_tenant"


def test_defaults_applied():
    cfg = TenantsConfig.model_validate(_minimal())
    assert cfg.defaults.vad.silence_threshold_ms == 600
    assert cfg.defaults.interruption.strategy == "min_words"


# ── Invalid configs ────────────────────────────────────────────────────────────

def test_missing_default_route_fails():
    d = _minimal()
    d["tenants"]["acme"]["routes"] = {
        "1000": {"pipeline": "my_pipeline", "context": "ctx"}
        # no "default"
    }
    with pytest.raises(ValidationError, match="missing 'default' route"):
        TenantsConfig.model_validate(d)


def test_pipeline_references_unknown_provider_fails():
    d = _minimal()
    d["pipelines"]["bad"] = {
        "kind": "modular",
        "stt": "nonexistent_stt",
        "llm": "llm_p",
        "tts": "tts_p",
    }
    d["tenants"]["acme"]["routes"]["1000"] = {"pipeline": "bad", "context": "ctx"}
    with pytest.raises(ValidationError, match="not in providers"):
        TenantsConfig.model_validate(d)


def test_pipeline_wrong_provider_mode_fails():
    d = _minimal()
    # stt_p has mode=stt; using it as llm should fail
    d["pipelines"]["bad"] = {
        "kind": "modular",
        "stt": "stt_p",
        "llm": "stt_p",   # wrong: stt provider used as llm
        "tts": "tts_p",
    }
    d["tenants"]["acme"]["routes"]["1000"] = {"pipeline": "bad", "context": "ctx"}
    with pytest.raises(ValidationError, match="expected 'llm'"):
        TenantsConfig.model_validate(d)


def test_route_references_unknown_pipeline_fails():
    d = _minimal()
    d["tenants"]["acme"]["routes"]["default"] = {
        "pipeline": "no_such_pipeline",
        "context": "ctx",
    }
    with pytest.raises(ValidationError, match="not in pipelines"):
        TenantsConfig.model_validate(d)


def test_route_references_unknown_context_fails():
    d = _minimal()
    d["tenants"]["acme"]["routes"]["default"] = {
        "pipeline": "my_pipeline",
        "context": "no_such_ctx",
    }
    with pytest.raises(ValidationError, match="not in contexts"):
        TenantsConfig.model_validate(d)


def test_fallback_use_tenant_unknown_tenant_fails():
    d = _minimal()
    d["fallback"] = {"policy": "use_tenant", "tenant": "no_such_tenant"}
    with pytest.raises(ValidationError, match="not in tenants"):
        TenantsConfig.model_validate(d)


def test_full_agent_pipeline_wrong_mode_fails():
    d = _minimal()
    # tts_p has mode=tts, not full_agent
    d["pipelines"]["fa_bad"] = {"kind": "full_agent", "provider": "tts_p"}
    d["tenants"]["acme"]["routes"]["1002"] = {"pipeline": "fa_bad", "context": "ctx"}
    with pytest.raises(ValidationError, match="expected 'full_agent'"):
        TenantsConfig.model_validate(d)
