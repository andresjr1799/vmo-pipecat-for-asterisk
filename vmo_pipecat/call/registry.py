"""
CallRegistry — in-memory registry of all active calls.

Provides async-safe lookup by vmo_call_id or asterisk_channel_id.
Used by /admin/calls/active (Phase 10) and defensive cleanup.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .controller import CallController


class CallRegistry:
    """Thread/asyncio-safe map of active CallControllers."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._by_vmo_call_id: dict[str, "CallController"] = {}
        self._by_channel: dict[str, "CallController"] = {}

    async def add(self, controller: "CallController") -> None:
        identity = controller.identity
        async with self._lock:
            self._by_vmo_call_id[identity.vmo_call_id] = controller
            self._by_channel[identity.asterisk_channel_id] = controller

    async def remove(self, controller: "CallController") -> None:
        identity = controller.identity
        async with self._lock:
            self._by_vmo_call_id.pop(identity.vmo_call_id, None)
            self._by_channel.pop(identity.asterisk_channel_id, None)

    def get_by_vmo_call_id(self, vmo_call_id: str) -> Optional["CallController"]:
        return self._by_vmo_call_id.get(vmo_call_id)

    def get_by_channel(self, asterisk_channel_id: str) -> Optional["CallController"]:
        return self._by_channel.get(asterisk_channel_id)

    @property
    def active_count(self) -> int:
        return len(self._by_vmo_call_id)

    def all_controllers(self) -> list["CallController"]:
        return list(self._by_vmo_call_id.values())

    def summary(self) -> list[dict]:
        return [
            {
                "vmo_call_id": c.identity.vmo_call_id,
                "tenant_id": c.identity.tenant_id,
                "did": c.identity.did,
                "node_id": c.identity.node_id,
            }
            for c in self._by_vmo_call_id.values()
        ]
