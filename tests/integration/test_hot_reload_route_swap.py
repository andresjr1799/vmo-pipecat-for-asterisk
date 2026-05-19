"""
Integration test — hot-reload route swap (Phase 10).

Verifies §7.2 (session isolation):
  - Editing tenants.yaml changes the route for NEW calls.
  - Active calls keep their frozen SessionConfig (immune to reload).
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, AsyncMock

import pytest

from vmo_pipecat.config.loader import load_tenants_config
from vmo_pipecat.config.models import (
    AudioProfileCfg,
    ContextCfg,
    DeepgramProviderCfg,
    ElevenLabsProviderCfg,
    ModularPipelineCfg,
    OpenAIProviderCfg,
    TransferCfg,
    OverridesCfg,
)
from vmo_pipecat.config.store import ConfigStore
from vmo_pipecat.tenancy.resolver import TenantResolver, SessionConfig


# ── YAML builder ───────────────────────────────────────────────────────────────

def _yaml_v1(tmp_path) -> str:
    """Initial config: DID 1000 → pipeline_a."""
    content = """
audio_profiles:
  telephony_8k:
    in_rate: 8000
    out_rate: 8000
    codec: slin
    channels: 1
providers:
  stt_a: {kind: deepgram, mode: stt, api_key: "dk_a"}
  stt_b: {kind: deepgram, mode: stt, api_key: "dk_b"}
  llm_p: {kind: openai, mode: llm, api_key: "ok"}
  tts_p: {kind: elevenlabs, mode: tts, api_key: "ek"}
pipelines:
  pipeline_a:
    kind: modular
    stt: stt_a
    llm: llm_p
    tts: tts_p
  pipeline_b:
    kind: modular
    stt: stt_b
    llm: llm_p
    tts: tts_p
contexts:
  ctx_a: {prompt: "Context A", audio_profile: telephony_8k}
  ctx_b: {prompt: "Context B", audio_profile: telephony_8k}
tenants:
  acme:
    name: Acme
    routes:
      "1000": {pipeline: pipeline_a, context: ctx_a}
      default: {pipeline: pipeline_a, context: ctx_a}
"""
    p = tmp_path / "tenants.yaml"
    p.write_text(content)
    return str(p)


def _yaml_v2(tmp_path) -> str:
    """After reload: DID 1000 → pipeline_b."""
    content = """
audio_profiles:
  telephony_8k:
    in_rate: 8000
    out_rate: 8000
    codec: slin
    channels: 1
providers:
  stt_a: {kind: deepgram, mode: stt, api_key: "dk_a"}
  stt_b: {kind: deepgram, mode: stt, api_key: "dk_b"}
  llm_p: {kind: openai, mode: llm, api_key: "ok"}
  tts_p: {kind: elevenlabs, mode: tts, api_key: "ek"}
pipelines:
  pipeline_a:
    kind: modular
    stt: stt_a
    llm: llm_p
    tts: tts_p
  pipeline_b:
    kind: modular
    stt: stt_b
    llm: llm_p
    tts: tts_p
contexts:
  ctx_a: {prompt: "Context A", audio_profile: telephony_8k}
  ctx_b: {prompt: "Context B", audio_profile: telephony_8k}
tenants:
  acme:
    name: Acme
    routes:
      "1000": {pipeline: pipeline_b, context: ctx_b}
      default: {pipeline: pipeline_b, context: ctx_b}
"""
    p = tmp_path / "tenants.yaml"
    p.write_text(content)
    return str(p)


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_new_call_after_reload_uses_new_route(tmp_path):
    """After hot-reload, resolve() for the same DID returns the new pipeline."""
    store = ConfigStore()
    resolver = TenantResolver(store)

    # v1: DID 1000 → pipeline_a / ctx_a
    path = _yaml_v1(tmp_path)
    cfg1, sha1 = load_tenants_config(path)
    store.swap(cfg1, source_path=path, sha256=sha1)
    v1_version = store.version

    sc_v1 = resolver.resolve("acme", "1000")
    assert sc_v1.pipeline.stt == "stt_a"
    assert sc_v1.context.prompt == "Context A"
    assert sc_v1.config_version == v1_version

    # Simulate hot-reload: v2 — DID 1000 → pipeline_b / ctx_b
    path2 = _yaml_v2(tmp_path)
    cfg2, sha2 = load_tenants_config(path2)
    store.swap(cfg2, source_path=path2, sha256=sha2)
    v2_version = store.version
    assert v2_version == v1_version + 1

    sc_v2 = resolver.resolve("acme", "1000")
    assert sc_v2.pipeline.stt == "stt_b"
    assert sc_v2.context.prompt == "Context B"
    assert sc_v2.config_version == v2_version


def test_active_call_frozen_config_immune_to_reload(tmp_path):
    """A SessionConfig captured before reload must not change after reload."""
    store = ConfigStore()
    resolver = TenantResolver(store)

    path = _yaml_v1(tmp_path)
    cfg1, sha1 = load_tenants_config(path)
    store.swap(cfg1, source_path=path, sha256=sha1)

    # Capture SessionConfig at call start (simulates StasisStart)
    frozen = resolver.resolve("acme", "1000")
    assert frozen.pipeline.stt == "stt_a"
    captured_version = frozen.config_version

    # Hot-reload changes the route
    path2 = _yaml_v2(tmp_path)
    cfg2, sha2 = load_tenants_config(path2)
    store.swap(cfg2, source_path=path2, sha256=sha2)

    # The frozen config is unchanged — isolation guarantee (§7.2)
    assert frozen.pipeline.stt == "stt_a", "Frozen config must be immutable after reload"
    assert frozen.context.prompt == "Context A"
    assert frozen.config_version == captured_version   # version from time of capture

    # But the store has advanced
    assert store.version == captured_version + 1


def test_configstore_version_increments_on_each_reload(tmp_path):
    store = ConfigStore()
    path = _yaml_v1(tmp_path)

    for i in range(1, 4):
        cfg, sha = load_tenants_config(path)
        store.swap(cfg, source_path=path, sha256=sha)
        assert store.version == i


def test_config_sha256_changes_on_file_change(tmp_path):
    store = ConfigStore()

    path1 = _yaml_v1(tmp_path)
    cfg1, sha1 = load_tenants_config(path1)
    store.swap(cfg1, source_path=path1, sha256=sha1)

    path2 = _yaml_v2(tmp_path)
    cfg2, sha2 = load_tenants_config(path2)
    store.swap(cfg2, source_path=path2, sha256=sha2)

    assert sha1 != sha2
    assert store.sha256 == sha2


def test_is_valid_false_after_mark_invalid(tmp_path):
    store = ConfigStore()
    path = _yaml_v1(tmp_path)
    cfg, sha = load_tenants_config(path)
    store.swap(cfg, source_path=path, sha256=sha)
    assert store.is_valid

    store.mark_invalid()
    assert not store.is_valid


def test_reload_restores_valid_after_mark_invalid(tmp_path):
    store = ConfigStore()
    path = _yaml_v1(tmp_path)
    cfg, sha = load_tenants_config(path)
    store.swap(cfg, source_path=path, sha256=sha)

    store.mark_invalid()
    assert not store.is_valid

    # Successful reload restores validity
    cfg2, sha2 = load_tenants_config(path)
    store.swap(cfg2, source_path=path, sha256=sha2)
    assert store.is_valid
