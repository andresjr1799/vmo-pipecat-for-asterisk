"""
ARIPool — multi-node Asterisk ARI client manager.

Manages one ARIClient per Asterisk node. Broadcasts events to registered
handlers after enriching each event with the source node_id (key:
``_vmo_node_id``). Callers can retrieve the specific client for a call via
``client_for(node_id)`` to route per-call ARI operations to the correct node.

Usage::

    pool = ARIPool()
    pool.add_node("ast-1", ARIClient(..., node_id="ast-1"))
    pool.add_node("ast-2", ARIClient(..., node_id="ast-2"))
    pool.on_event("StasisStart", engine._handle_stasis_start)
    await pool.start_all_listening()

Each handler receives the enriched event dict; ``event["_vmo_node_id"]`` is
the node that originated it.
"""

from __future__ import annotations

import asyncio
from typing import Callable, Dict, List, Optional

from ..observability.log_setup import get_logger
from .client import ARIClient
from .events import NODE_ID_KEY

logger = get_logger(__name__)


class ARIPool:
    """
    Manages a set of ARIClient instances keyed by node_id.

    Thread / concurrency model: ARIPool is used from a single asyncio event
    loop. ``add_node`` / ``remove_node`` may be called from any coroutine
    running on that loop.
    """

    def __init__(self) -> None:
        self._nodes: Dict[str, ARIClient] = {}
        # Pool-level handlers registered before nodes exist
        self._handlers: Dict[str, List[Callable]] = {}
        # Listener tasks keyed by node_id
        self._listener_tasks: Dict[str, asyncio.Task] = {}

    # ------------------------------------------------------------------
    # Node management
    # ------------------------------------------------------------------

    def add_node(self, node_id: str, client: ARIClient) -> None:
        """Register an ARIClient for the given node_id.

        Re-registers all already-known pool-level event handlers on the new
        client so that a node added after ``on_event()`` calls still receives
        all registrations.
        """
        self._nodes[node_id] = client
        for event_type, handlers in self._handlers.items():
            for handler in handlers:
                self._bind_node_handler(node_id, client, event_type, handler)
        logger.info("ARIPool: node added", node_id=node_id)

    def remove_node(self, node_id: str) -> Optional[ARIClient]:
        """Deregister a node. Returns the removed ARIClient or None."""
        client = self._nodes.pop(node_id, None)
        task = self._listener_tasks.pop(node_id, None)
        if task and not task.done():
            task.cancel()
        if client:
            logger.info("ARIPool: node removed", node_id=node_id)
        return client

    # ------------------------------------------------------------------
    # Event routing
    # ------------------------------------------------------------------

    def on_event(self, event_type: str, handler: Callable) -> None:
        """Register a pool-level event handler for all nodes.

        The handler receives the event dict with ``_vmo_node_id`` injected.
        Equivalent to calling ``add_event_handler`` on every ARIClient in the
        pool, now and in the future.
        """
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
        # Bind to all currently known nodes
        for node_id, client in self._nodes.items():
            self._bind_node_handler(node_id, client, event_type, handler)

    def _bind_node_handler(
        self,
        node_id: str,
        client: ARIClient,
        event_type: str,
        pool_handler: Callable,
    ) -> None:
        """Register a wrapper on ``client`` that injects node_id before calling
        ``pool_handler``."""

        async def enriched_handler(
            event: dict,
            _nid: str = node_id,
            _h: Callable = pool_handler,
        ) -> None:
            event[NODE_ID_KEY] = _nid
            await _h(event)

        # Preserve the original handler name for introspection
        enriched_handler.__name__ = f"{pool_handler.__name__}@{node_id}"
        client.add_event_handler(event_type, enriched_handler)

    # ------------------------------------------------------------------
    # Client access
    # ------------------------------------------------------------------

    def client_for(self, node_id: str) -> ARIClient:
        """Return the ARIClient for the given node_id.

        Raises KeyError if node_id is unknown — callers should validate first.
        """
        return self._nodes[node_id]

    @property
    def default_node_id(self) -> Optional[str]:
        """The node_id of the first registered node, or None if pool is empty."""
        try:
            return next(iter(self._nodes))
        except StopIteration:
            return None

    @property
    def default_client(self) -> Optional[ARIClient]:
        """The ARIClient of the first registered node (single-node compat)."""
        nid = self.default_node_id
        return self._nodes[nid] if nid else None

    @property
    def node_ids(self) -> List[str]:
        return list(self._nodes.keys())

    @property
    def is_any_connected(self) -> bool:
        return any(c.is_connected for c in self._nodes.values())

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start_all_listening(self) -> None:
        """Start supervised listener tasks for all registered nodes.

        Creates one asyncio.Task per node. Already-running tasks are skipped.
        """
        for node_id, client in self._nodes.items():
            if node_id in self._listener_tasks and not self._listener_tasks[node_id].done():
                continue
            task = asyncio.create_task(
                client.start_listening(), name=f"ari-listener-{node_id}"
            )
            self._listener_tasks[node_id] = task
            logger.info("ARIPool: started listener task", node_id=node_id)

    async def stop_all(self) -> None:
        """Disconnect all nodes and cancel listener tasks."""
        for node_id, client in list(self._nodes.items()):
            task = self._listener_tasks.pop(node_id, None)
            if task and not task.done():
                task.cancel()
            try:
                await client.disconnect()
            except Exception as exc:
                logger.warning(
                    "ARIPool: error disconnecting node",
                    node_id=node_id,
                    error=str(exc),
                )
        logger.info("ARIPool: all nodes stopped")

    async def reload(self, new_node_configs: list) -> None:
        """Hot-reload node list from a new list of AsteriskNodeConfig objects.

        Adds nodes that are new, removes nodes that disappeared.
        Existing nodes whose credentials/host haven't changed are kept as-is
        (their live WS connections survive the reload).

        Args:
            new_node_configs: list of AsteriskNodeConfig (Pydantic models from
                vmo_engine.config.models).
        """
        from ..config.models import AsteriskNodeConfig

        new_ids = {cfg.id for cfg in new_node_configs}
        existing_ids = set(self._nodes.keys())

        # Remove disappeared nodes
        for nid in existing_ids - new_ids:
            client = self.remove_node(nid)
            if client:
                await client.disconnect()

        # Add new nodes
        for cfg in new_node_configs:
            if cfg.id not in self._nodes:
                client = _make_client_from_config(cfg)
                self.add_node(cfg.id, client)
                asyncio.create_task(
                    client.start_listening(), name=f"ari-listener-{cfg.id}"
                )
                self._listener_tasks[cfg.id] = asyncio.current_task()  # overwritten below

        logger.info(
            "ARIPool: reload complete",
            added=list(new_ids - existing_ids),
            removed=list(existing_ids - new_ids),
        )


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def _make_client_from_config(cfg: "AsteriskNodeConfig") -> ARIClient:
    """Build an ARIClient from an AsteriskNodeConfig."""
    scheme = getattr(cfg, "ari_scheme", "http") or "http"
    host = cfg.host
    port = getattr(cfg, "ari_port", 8088) or 8088
    base_url = f"{scheme}://{host}:{port}/ari"
    return ARIClient(
        username=cfg.ari_username,
        password=cfg.ari_password,
        base_url=base_url,
        app_name=cfg.ari_app,
        node_id=cfg.id,
    )
