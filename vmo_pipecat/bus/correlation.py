"""
CorrelationRouter — routes inbound bus messages to the right asyncio.Queue per turn.

Each LLM turn gets a unique correlation_id = f"{vmo_call_id}:{turn_id}".
Inbound messages are matched by correlation_id and delivered to the waiting
consumer. Late messages (correlation_id no longer registered) are silently
discarded with an info log.
"""

from __future__ import annotations

import asyncio
from typing import Optional

from ..observability.log_setup import get_logger

logger = get_logger(__name__)

# Sentinel value that signals a closed (cancelled) correlation slot.
_CLOSED = object()


class CorrelationRouter:
    """Maps correlation_id → asyncio.Queue[dict | None] per active LLM turn.

    Thread-safety: async-safe (all access from single event loop).
    """

    def __init__(self) -> None:
        self._queues: dict[str, asyncio.Queue] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, correlation_id: str) -> asyncio.Queue:
        """Register a new turn and return its response queue.

        The queue yields parsed response dicts (one per chunk/final/end/error).
        A `None` sentinel marks the end of the stream (closed/cancelled).
        """
        q: asyncio.Queue = asyncio.Queue()
        self._queues[correlation_id] = q
        logger.debug("CorrelationRouter: registered", correlation_id=correlation_id)
        return q

    # ------------------------------------------------------------------
    # Delivery (called by the inbound subscription handler)
    # ------------------------------------------------------------------

    async def deliver(self, message: dict) -> bool:
        """Deliver a parsed response message to the waiting consumer.

        Returns True if delivered, False if correlation_id was unknown (late delivery).
        """
        cid = message.get("correlation_id", "")
        q = self._queues.get(cid)
        if q is None:
            logger.info(
                "CorrelationRouter: late/unknown correlation_id — discarding",
                correlation_id=cid,
                msg_type=message.get("type"),
            )
            return False
        await q.put(message)
        return True

    # ------------------------------------------------------------------
    # Cleanup / cancel
    # ------------------------------------------------------------------

    def close(self, correlation_id: str) -> None:
        """Close a turn's queue, unblocking any consumer awaiting it."""
        q = self._queues.pop(correlation_id, None)
        if q is not None:
            try:
                q.put_nowait(None)   # sentinel → consumer exits its loop
            except asyncio.QueueFull:
                pass
            logger.debug("CorrelationRouter: closed", correlation_id=correlation_id)

    def discard_duplicates(self, correlation_id: str, message: dict) -> bool:
        """Check if a message with this correlation_id is still expected.

        Re-deliveries for an already-closed turn return False.
        """
        return correlation_id in self._queues

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def active_count(self) -> int:
        return len(self._queues)

    def active_ids(self) -> list[str]:
        return list(self._queues.keys())
