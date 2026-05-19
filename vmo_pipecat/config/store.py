"""
ConfigStore — thread-safe holder of the current TenantsConfig snapshot.

Callers read `store.current` at call-start time and freeze it into a
SessionConfig.  No component reads ConfigStore.current during an active call.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Optional

from .models import TenantsConfig


class ConfigStore:
    """Atomic, thread-safe store for the active TenantsConfig snapshot."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._config: Optional[TenantsConfig] = None
        self._version: int = 0
        self._loaded_at: Optional[datetime] = None
        self._source_path: str = ""
        self._sha256: str = ""
        self._valid: bool = False

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    @property
    def current(self) -> Optional[TenantsConfig]:
        with self._lock:
            return self._config

    @property
    def version(self) -> int:
        with self._lock:
            return self._version

    @property
    def loaded_at_iso(self) -> str:
        with self._lock:
            return self._loaded_at.isoformat() if self._loaded_at else ""

    @property
    def source_path(self) -> str:
        with self._lock:
            return self._source_path

    @property
    def sha256(self) -> str:
        with self._lock:
            return self._sha256

    @property
    def is_valid(self) -> bool:
        with self._lock:
            return self._valid

    def info(self) -> dict:
        with self._lock:
            return {
                "config_version": self._version,
                "loaded_at_iso": self._loaded_at.isoformat() if self._loaded_at else None,
                "source_path": self._source_path,
                "sha256": self._sha256,
                "valid": self._valid,
            }

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def swap(
        self,
        config: TenantsConfig,
        source_path: str = "",
        sha256: str = "",
    ) -> None:
        """Atomically replace the current snapshot."""
        with self._lock:
            self._config = config
            self._version += 1
            self._loaded_at = datetime.now(tz=timezone.utc)
            self._source_path = source_path
            self._sha256 = sha256
            self._valid = True

    def mark_invalid(self) -> None:
        """Mark config as degraded (last reload failed validation)."""
        with self._lock:
            self._valid = False
