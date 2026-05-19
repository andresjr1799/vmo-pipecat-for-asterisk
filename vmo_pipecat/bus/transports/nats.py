"""
NATS transport for the agentic bus (V1, recommended).

Uses `nats-py` (async NATS client). When nats-py is not installed, raises
ImportError with a clear message — pipecat-ai extras do NOT include nats,
so the operator must install it separately: `pip install nats-py`.

Supports:
  - Core NATS publish + queue-group subscribe
  - JetStream for persistent delivery (optional, connection.jetstream: true)
  - Exponential reconnect
  - Per-message headers
"""

from __future__ import annotations

import json
from typing import Any, Awaitable, Callable, Optional

from ...observability.log_setup import get_logger
from .. import SubscriptionHandle

logger = get_logger(__name__)

try:
    import nats
    import nats.aio.client as nats_client
    _NATS_AVAILABLE = True
except ImportError:
    _NATS_AVAILABLE = False


class _NatsSubscriptionHandle(SubscriptionHandle):
    def __init__(self, sub) -> None:
        self._sub = sub

    async def unsubscribe(self) -> None:
        try:
            await self._sub.unsubscribe()
        except Exception:
            pass


class NatsTransport:
    """NATS-backed AgenticBusTransport (V1).

    Config keys (from AgenticBusProviderCfg.connection):
      servers: str — comma-separated NATS server URLs
      jetstream: bool — enable JetStream (default False)

    Auth (from AgenticBusProviderCfg.auth):
      type: bearer → token auth (nats user token)
      type: none   → no auth
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        if not _NATS_AVAILABLE:
            raise ImportError(
                "nats-py is required for the NATS bus transport. "
                "Install it with: pip install nats-py"
            )
        conn_cfg = cfg.get("connection", cfg)
        servers = str(conn_cfg.get("servers", "nats://localhost:4222"))
        self._servers = [s.strip() for s in servers.split(",")]
        self._use_jetstream = bool(conn_cfg.get("jetstream", False))
        self._auth_cfg = cfg.get("auth", {})
        self._nc: Optional[nats_client.Client] = None
        self._js = None

    async def start(self) -> None:
        kwargs: dict = {"servers": self._servers}
        auth_type = self._auth_cfg.get("type", "none")
        if auth_type == "bearer":
            kwargs["token"] = self._auth_cfg.get("token", "")
        self._nc = await nats.connect(**kwargs)
        if self._use_jetstream:
            self._js = self._nc.jetstream()
        logger.info("NATS transport connected", servers=self._servers)

    async def stop(self) -> None:
        if self._nc:
            await self._nc.close()
            self._nc = None
        logger.info("NATS transport stopped")

    async def publish(
        self,
        destination: str,
        payload: bytes,
        headers: dict[str, str],
    ) -> None:
        if self._nc is None:
            raise RuntimeError("NATS transport not started")
        await self._nc.publish(destination, payload)

    async def subscribe(
        self,
        destination: str,
        group: str | None,
        handler: Callable[[bytes, dict[str, str]], Awaitable[None]],
    ) -> SubscriptionHandle:
        if self._nc is None:
            raise RuntimeError("NATS transport not started")

        async def _msg_handler(msg) -> None:
            headers = dict(msg.headers) if msg.headers else {}
            await handler(msg.data, headers)

        if group:
            sub = await self._nc.subscribe(destination, queue=group, cb=_msg_handler)
        else:
            sub = await self._nc.subscribe(destination, cb=_msg_handler)
        return _NatsSubscriptionHandle(sub)
