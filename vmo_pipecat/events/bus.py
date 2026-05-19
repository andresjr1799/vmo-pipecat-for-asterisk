"""
EventBus — bidireccional Asterisk ↔ PipeCat event bus.

Interface (Protocol) + LoggingEventBus (default, Day-1 implementation).
Every payload is auto-enriched with the call identity before emission.

NatsEventBus stub lives in nats_bus.py (Phase 12).
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Optional, Protocol, runtime_checkable, TYPE_CHECKING

if TYPE_CHECKING:
    from ..call.identity import CallIdentity


@runtime_checkable
class EventBus(Protocol):
    async def emit(
        self,
        subject: str,
        identity: Optional["CallIdentity"] = None,
        **payload,
    ) -> None: ...

    async def close(self) -> None: ...


class LoggingEventBus:
    """Default EventBus: writes JSON events to stdout with prefix 'EVENT'.

    Docker → Loki/CloudWatch collects them.  Zero external dependencies.
    Each event includes full call identity + ISO timestamp.
    """

    async def emit(
        self,
        subject: str,
        identity: Optional["CallIdentity"] = None,
        **payload,
    ) -> None:
        data: dict = {
            "subject": subject,
            "ts": datetime.now(tz=timezone.utc).isoformat(),
        }
        if identity is not None:
            data.update(identity.asdict())
        data.update(payload)
        print(f"EVENT {json.dumps(data, default=str)}", flush=True)

    async def close(self) -> None:
        pass
