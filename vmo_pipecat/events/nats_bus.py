"""NatsEventBus — stub for Phase 12. Not used until VMO_EVENT_BUS=nats."""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..call.identity import CallIdentity


class NatsEventBus:
    """Publishes call events to NATS subjects. Implemented in Phase 12."""

    async def emit(
        self,
        subject: str,
        identity: Optional["CallIdentity"] = None,
        **payload,
    ) -> None:
        raise NotImplementedError("NatsEventBus is implemented in Phase 12")

    async def close(self) -> None:
        pass
