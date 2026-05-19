"""
CallRouter — dispatches AudioSocket events to the correct CallController.

Maps:
  audio_uuid (str UUID) → pending controller (before UUID handshake)
  conn_id    (hex str)  → active controller  (after handshake)
  vmo_call_id           → controller         (for ARI event lookup)
  asterisk_channel_id   → controller         (for ARI event lookup)

Thread safety: uses asyncio.Lock — all public methods are async coroutines
called from the single event loop.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

from ..observability.log_setup import get_logger

logger = get_logger(__name__)


class CallRouter:
    """Routes AudioSocket events (audio, DTMF, disconnect) to controllers.

    The controller duck type must expose:
        async bind_audio_conn(conn_id: str) -> None
        async on_audio(audio: bytes) -> None
        async on_dtmf(digit: str) -> None
        async on_disconnect() -> None
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        # Registered but not yet bound (UUID handshake pending)
        self._pending: dict[str, Any] = {}            # audio_uuid → controller
        # Active connections
        self._by_conn: dict[str, Any] = {}            # conn_id → controller
        # Secondary lookup keys (registered by CallLifecycle in Phase 3)
        self._by_vmo_call_id: dict[str, Any] = {}    # vmo_call_id → controller
        self._by_channel: dict[str, Any] = {}         # asterisk_channel_id → controller

    # ------------------------------------------------------------------
    # Registration (called by CallLifecycle before originating AudioSocket)
    # ------------------------------------------------------------------

    async def register_pending(self, audio_uuid: str, controller: Any) -> None:
        """Register a controller waiting for an AudioSocket UUID handshake."""
        async with self._lock:
            self._pending[str(audio_uuid)] = controller
        logger.debug("CallRouter: pending registered", audio_uuid=audio_uuid)

    def register_by_vmo_call_id(self, vmo_call_id: str, controller: Any) -> None:
        self._by_vmo_call_id[str(vmo_call_id)] = controller

    def register_by_channel(self, asterisk_channel_id: str, controller: Any) -> None:
        self._by_channel[str(asterisk_channel_id)] = controller

    # ------------------------------------------------------------------
    # UUID handshake (called by AudioSocketServer.on_uuid)
    # ------------------------------------------------------------------

    async def bind_uuid(self, audio_uuid: str, conn_id: str) -> bool:
        """Called when AudioSocket receives the TYPE_UUID frame.

        Moves the controller from pending → active (keyed by conn_id).
        Returns True to accept the connection, False to reject.
        """
        async with self._lock:
            controller = self._pending.pop(str(audio_uuid), None)
            if controller is None:
                logger.warning(
                    "CallRouter: unknown audio_uuid in handshake",
                    audio_uuid=audio_uuid,
                    conn_id=conn_id,
                )
                return False
            self._by_conn[conn_id] = controller

        logger.info(
            "CallRouter: UUID bound",
            audio_uuid=audio_uuid,
            conn_id=conn_id,
        )
        await controller.bind_audio_conn(conn_id)
        return True

    # ------------------------------------------------------------------
    # Audio / DTMF / disconnect dispatch
    # ------------------------------------------------------------------

    async def dispatch_audio(self, conn_id: str, audio: bytes) -> None:
        async with self._lock:
            controller = self._by_conn.get(conn_id)
        if controller:
            await controller.on_audio(audio)

    async def dispatch_dtmf(self, conn_id: str, digit: str) -> None:
        async with self._lock:
            controller = self._by_conn.get(conn_id)
        if controller:
            await controller.on_dtmf(digit)

    async def dispatch_disconnect(self, conn_id: str) -> None:
        async with self._lock:
            controller = self._by_conn.pop(conn_id, None)
        if controller:
            logger.info("CallRouter: disconnect dispatched", conn_id=conn_id)
            await controller.on_disconnect()

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get_by_vmo_call_id(self, vmo_call_id: str) -> Optional[Any]:
        return self._by_vmo_call_id.get(str(vmo_call_id))

    def get_by_channel(self, asterisk_channel_id: str) -> Optional[Any]:
        return self._by_channel.get(str(asterisk_channel_id))

    def get_by_conn(self, conn_id: str) -> Optional[Any]:
        return self._by_conn.get(conn_id)

    # ------------------------------------------------------------------
    # Cleanup (called by CallController.shutdown)
    # ------------------------------------------------------------------

    def remove(self, controller: Any) -> None:
        """Remove all references to a controller (any map)."""
        self._by_vmo_call_id = {k: v for k, v in self._by_vmo_call_id.items() if v is not controller}
        self._by_channel = {k: v for k, v in self._by_channel.items() if v is not controller}
        # _by_conn is cleaned up via dispatch_disconnect
        # _pending entries expire naturally when UUID is received or call is abandoned

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def active_count(self) -> int:
        return len(self._by_conn)

    @property
    def pending_count(self) -> int:
        return len(self._pending)
