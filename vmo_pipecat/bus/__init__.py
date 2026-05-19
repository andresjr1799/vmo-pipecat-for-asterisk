"""
vmo_pipecat.bus — agentic bus client (§9.5).

The bus connects vmo-pipecat-for-asterisk to an external backend agentic system
(LangChain, LangGraph, or any other orchestrator) via a message broker.

This package is CLIENT-ONLY. The broker and backend are external infrastructure.

Transport pluggability (§9.5.5):
  transport: nats       → NatsTransport (V1, recommended)
  transport: rabbitmq   → RabbitMQTransport (optional V1.x)
  transport: kinesis    → KinesisTransport (optional V1.x)
  transport: kafka      → KafkaTransport (optional V1.x)
  transport: sqs        → SQSTransport (optional V1.x)
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Protocol, runtime_checkable


# ── Protocol ───────────────────────────────────────────────────────────────────

class SubscriptionHandle:
    """Handle returned by subscribe(); call unsubscribe() to cancel."""
    async def unsubscribe(self) -> None:
        pass


@runtime_checkable
class AgenticBusTransport(Protocol):
    """Common interface for all bus transport implementations."""

    async def start(self) -> None: ...
    async def stop(self) -> None: ...

    async def publish(
        self,
        destination: str,
        payload: bytes,
        headers: dict[str, str],
    ) -> None: ...

    async def subscribe(
        self,
        destination: str,
        group: str | None,
        handler: Callable[[bytes, dict[str, str]], Awaitable[None]],
    ) -> SubscriptionHandle: ...


# ── Factory ────────────────────────────────────────────────────────────────────

def make_bus_transport(kind: str, cfg: dict[str, Any]) -> AgenticBusTransport:
    """Instantiate the correct transport for the given `transport` kind."""
    if kind == "nats":
        from .transports.nats import NatsTransport
        return NatsTransport(cfg)
    if kind in ("rabbitmq", "kinesis", "kafka", "sqs"):
        raise ImportError(
            f"Transport '{kind}' is not yet implemented in V1. "
            f"Contribute it at vmo_pipecat/bus/transports/{kind}.py."
        )
    raise ValueError(
        f"Unknown bus transport kind: '{kind}'. Supported in V1: nats"
    )
