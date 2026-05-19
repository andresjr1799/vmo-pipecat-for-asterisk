"""
tenants.yaml loader.

Reads the YAML file, expands ${VAR} and ${VAR:-default} env-var references,
validates with Pydantic v2, and returns TenantsConfig + sha256 of the raw file.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any

import yaml

from .models import TenantsConfig

# Matches ${VAR} and ${VAR:-default}
_ENV_RE = re.compile(r"\$\{([^}:]+)(?::-(.*?))?\}")


def _expand_env(text: str) -> str:
    """Replace ${VAR} / ${VAR:-default} in a YAML string before parsing."""
    def _sub(m: re.Match) -> str:
        var_name = m.group(1)
        default = m.group(2) if m.group(2) is not None else ""
        return os.environ.get(var_name, default)

    return _ENV_RE.sub(_sub, text)


def load_tenants_config(path: str | Path) -> tuple[TenantsConfig, str]:
    """Load and validate tenants.yaml.

    Returns:
        (TenantsConfig, sha256_hex) — sha256 is of the raw (unexpanded) file content.

    Raises:
        FileNotFoundError: if path does not exist.
        yaml.YAMLError: if YAML is malformed.
        pydantic.ValidationError: if schema validation fails.
    """
    raw = Path(path).read_text(encoding="utf-8")
    sha256 = hashlib.sha256(raw.encode()).hexdigest()

    expanded = _expand_env(raw)
    data = yaml.safe_load(expanded) or {}

    # Recursively replace None with "" so missing env vars don't fail validation
    data = _sanitize_none(data)

    config = TenantsConfig.model_validate(data)
    return config, sha256


def _sanitize_none(obj: Any) -> Any:
    """Recursively replace None with empty string in the parsed data tree."""
    if obj is None:
        return ""
    if isinstance(obj, dict):
        return {k: _sanitize_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_none(v) for v in obj]
    return obj
