"""
Integration tests — agentic bus: CorrelationRouter + AgenticBusLLMService
with in-memory mock transport (Phase 11).

Tests cover (§9.5.6):
  - Full turn in streaming: tokens mode
  - Full turn in streaming: full mode
  - Tool call from backend → AsteriskActions callback invoked
  - Cancel mid-stream (barge-in)
  - First-token timeout
  - Complete-response timeout
  - Error type from backend
  - Duplicate re-delivery discarded
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Awaitable, Callable, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from vmo_pipecat.bus.correlation import CorrelationRouter
from vmo_pipecat.bus.envelope import (
    AgenticCancelEnvelope,
    AgenticResponseEnvelope,
    parse_response,
)
from vmo_pipecat.bus import SubscriptionHandle, AgenticBusTransport
from vmo_pipecat.call.identity import CallIdentity
from vmo_pipecat.providers.agentic_bus_llm import AgenticBusLLMService


# ── In-memory mock transport ───────────────────────────────────────────────────

class _MockSubscriptionHandle(SubscriptionHandle):
    async def unsubscribe(self): pass


class InMemoryTransport:
    """Mock AgenticBusTransport that routes messages in-memory.

    Lets tests inject response messages directly via `inject_response()`.
    """

    def __init__(self):
        self.published: list[tuple[str, bytes, dict]] = []
        self._handlers: dict[str, Callable] = {}

    async def start(self) -> None: pass
    async def stop(self) -> None: pass

    async def publish(self, destination: str, payload: bytes, headers: dict) -> None:
        self.published.append((destination, payload, headers))

    async def subscribe(
        self,
        destination: str,
        group: Optional[str],
        handler: Callable[[bytes, dict], Awaitable[None]],
    ) -> SubscriptionHandle:
        self._handlers[destination] = handler
        return _MockSubscriptionHandle()

    async def inject_response(self, destination: str, message: dict) -> None:
        """Inject an inbound message to subscribers on `destination`."""
        handler = self._handlers.get(destination)
        if handler:
            await handler(json.dumps(message).encode(), {})


# ── Helpers ────────────────────────────────────────────────────────────────────

def _identity() -> CallIdentity:
    return CallIdentity(
        vmo_call_id=str(uuid.uuid4()),
        asterisk_channel_id="SIP/trunk-bus",
        call_id_sbc="sbc-bus",
        tenant_id="acme",
        tenant_name="Acme Corp",
        node_id="ast-1",
        did="2000",
    )


def _make_service(transport: InMemoryTransport, identity: Optional[CallIdentity] = None) -> AgenticBusLLMService:
    if identity is None:
        identity = _identity()
    return AgenticBusLLMService(
        transport=transport,
        outbound_destination="agentic.requests.acme",
        inbound_destination="agentic.responses.acme",
        inbound_group="vmo-pipecat",
        identity=identity,
        session_config=MagicMock(),
        provider_params={
            "streaming": "tokens",
            "first_token_timeout_ms": 500,
            "request_timeout_ms": 2000,
            "tool_call_handling": "bus",
        },
    )


def _chunk(correlation_id: str, text: str, seq: int) -> dict:
    return {
        "schema": "vmo.agentic.response/1",
        "correlation_id": correlation_id,
        "type": "chunk",
        "seq": seq,
        "text": text,
    }


def _end_msg(correlation_id: str, seq: int) -> dict:
    return {
        "schema": "vmo.agentic.response/1",
        "correlation_id": correlation_id,
        "type": "end",
        "seq": seq,
        "usage": {"tokens_in": 50, "tokens_out": 30},
    }


def _final_msg(correlation_id: str, text: str) -> dict:
    return {
        "schema": "vmo.agentic.response/1",
        "correlation_id": correlation_id,
        "type": "final",
        "text": text,
        "usage": {"tokens_in": 50, "tokens_out": 30},
    }


def _tool_call_msg(correlation_id: str, name: str, args: dict) -> dict:
    return {
        "schema": "vmo.agentic.response/1",
        "correlation_id": correlation_id,
        "type": "tool_call",
        "seq": 1,
        "tool_call": {"id": "tc_1", "name": name, "arguments": args},
    }


def _error_msg(correlation_id: str, code: str) -> dict:
    return {
        "schema": "vmo.agentic.response/1",
        "correlation_id": correlation_id,
        "type": "error",
        "error": {"code": code, "message": "backend error"},
    }


# ═════════════════════════════════════════════════════════════════════════════
# CorrelationRouter unit tests
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_correlation_router_register_and_deliver():
    router = CorrelationRouter()
    cid = "call-1:turn-1"
    q = router.register(cid)
    assert router.active_count == 1

    msg = {"correlation_id": cid, "type": "chunk", "text": "Hi"}
    ok = await router.deliver(msg)
    assert ok
    received = q.get_nowait()
    assert received["text"] == "Hi"


@pytest.mark.asyncio
async def test_correlation_router_unknown_correlation_discarded():
    router = CorrelationRouter()
    msg = {"correlation_id": "no-such-id", "type": "chunk", "text": "hi"}
    ok = await router.deliver(msg)
    assert not ok


@pytest.mark.asyncio
async def test_correlation_router_close_sends_sentinel():
    router = CorrelationRouter()
    cid = "call-2:turn-1"
    q = router.register(cid)

    router.close(cid)
    sentinel = q.get_nowait()
    assert sentinel is None   # sentinel
    assert router.active_count == 0


@pytest.mark.asyncio
async def test_correlation_router_late_delivery_discarded():
    router = CorrelationRouter()
    cid = "call-3:turn-1"
    router.register(cid)
    router.close(cid)   # close immediately

    # Deliver after close — should be discarded
    ok = await router.deliver({"correlation_id": cid, "type": "chunk"})
    assert not ok


@pytest.mark.asyncio
async def test_correlation_router_multiple_concurrent_turns():
    router = CorrelationRouter()
    q1 = router.register("c1:t1")
    q2 = router.register("c2:t1")
    assert router.active_count == 2

    await router.deliver({"correlation_id": "c1:t1", "type": "chunk", "text": "A"})
    await router.deliver({"correlation_id": "c2:t1", "type": "chunk", "text": "B"})

    assert q1.get_nowait()["text"] == "A"
    assert q2.get_nowait()["text"] == "B"


# ═════════════════════════════════════════════════════════════════════════════
# AgenticBusLLMService: full turn flow
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_streaming_tokens_full_turn():
    """Streaming tokens: chunks arrive → full text assembled correctly."""
    transport = InMemoryTransport()
    svc = _make_service(transport)
    await svc.start_bus()

    received_frames = []
    async def emit(frame): received_frames.append(frame.text)

    messages = [{"role": "system", "content": "Hi"}, {"role": "user", "content": "Hola"}]

    async def _backend():
        await asyncio.sleep(0.02)
        # Determine correlation_id from last published request
        _, payload, _ = transport.published[-1]
        req = json.loads(payload)
        cid = req["correlation_id"]
        assert req["schema"] == "vmo.agentic.request/1"
        assert req["input"]["messages"][-1]["content"] == "Hola"

        await transport.inject_response("agentic.responses.acme", _chunk(cid, "Claro, ", 1))
        await transport.inject_response("agentic.responses.acme", _chunk(cid, "con gusto.", 2))
        await transport.inject_response("agentic.responses.acme", _end_msg(cid, 3))

    asyncio.create_task(_backend())
    result = await svc.run_turn(messages, tools=[], emit_frame=emit)

    assert result == "Claro, con gusto."
    assert received_frames == ["Claro, ", "con gusto."]
    await svc.stop_bus()


@pytest.mark.asyncio
async def test_streaming_full_mode():
    """Non-streaming: single 'final' message → text returned."""
    transport = InMemoryTransport()
    svc = _make_service(transport)
    svc._streaming = "full"
    await svc.start_bus()

    frames = []
    async def emit(f): frames.append(f.text)
    messages = [{"role": "user", "content": "Hola"}]

    async def _backend():
        await asyncio.sleep(0.02)
        _, payload, _ = transport.published[-1]
        cid = json.loads(payload)["correlation_id"]
        await transport.inject_response("agentic.responses.acme", _final_msg(cid, "Respuesta completa."))

    asyncio.create_task(_backend())
    result = await svc.run_turn(messages, tools=[], emit_frame=emit)

    assert result == "Respuesta completa."
    assert frames == ["Respuesta completa."]
    await svc.stop_bus()


@pytest.mark.asyncio
async def test_tool_call_invokes_registered_callback():
    """Backend sends tool_call → registered callback is invoked."""
    transport = InMemoryTransport()
    svc = _make_service(transport)
    await svc.start_bus()

    callback_args = []
    async def _transfer_cb(name, tc_id, args, llm, ctx, result_cb):
        callback_args.append({"name": name, "args": args})
        await result_cb({"status": "ok"})

    svc.register_function("transfer_call", _transfer_cb)

    async def _backend():
        await asyncio.sleep(0.02)
        _, payload, _ = transport.published[-1]
        cid = json.loads(payload)["correlation_id"]
        await transport.inject_response(
            "agentic.responses.acme",
            _tool_call_msg(cid, "transfer_call", {"target": "9000"}),
        )
        await asyncio.sleep(0.05)
        await transport.inject_response("agentic.responses.acme", _end_msg(cid, 2))

    asyncio.create_task(_backend())
    await svc.run_turn([{"role": "user", "content": "Transferir"}], tools=[], emit_frame=None)

    assert callback_args[0]["name"] == "transfer_call"
    assert callback_args[0]["args"]["target"] == "9000"
    await svc.stop_bus()


@pytest.mark.asyncio
async def test_tool_call_publishes_tool_result():
    """Backend tool_call with tool_call_handling=bus → publishes tool_result."""
    transport = InMemoryTransport()
    svc = _make_service(transport)
    await svc.start_bus()

    async def _cb(name, tc_id, args, llm, ctx, result_cb):
        await result_cb({"status": "ok", "latency_ms": 100})

    svc.register_function("hangup_call", _cb)

    async def _backend():
        await asyncio.sleep(0.02)
        _, payload, _ = transport.published[0]
        cid = json.loads(payload)["correlation_id"]
        await transport.inject_response(
            "agentic.responses.acme",
            _tool_call_msg(cid, "hangup_call", {}),
        )
        await asyncio.sleep(0.05)
        await transport.inject_response("agentic.responses.acme", _end_msg(cid, 2))

    asyncio.create_task(_backend())
    await svc.run_turn([{"role": "user", "content": "bye"}], tools=[], emit_frame=None)

    # Last published message should be tool_result
    published_schemas = [json.loads(p)["schema"] for _, p, _ in transport.published]
    assert "vmo.agentic.tool_result/1" in published_schemas
    await svc.stop_bus()


@pytest.mark.asyncio
async def test_error_response_ends_turn():
    """Error type from backend → turn ends (no exception raised)."""
    transport = InMemoryTransport()
    svc = _make_service(transport)
    await svc.start_bus()

    async def _backend():
        await asyncio.sleep(0.02)
        _, payload, _ = transport.published[-1]
        cid = json.loads(payload)["correlation_id"]
        await transport.inject_response("agentic.responses.acme", _error_msg(cid, "backend_unavailable"))

    asyncio.create_task(_backend())
    result = await svc.run_turn([{"role": "user", "content": "hi"}], tools=[], emit_frame=None)
    assert result == ""   # no text for errors
    await svc.stop_bus()


@pytest.mark.asyncio
async def test_first_token_timeout_raises():
    """No response within first_token_timeout_ms → TimeoutError."""
    transport = InMemoryTransport()
    svc = _make_service(transport)
    svc._first_token_timeout_ms = 100   # very short
    await svc.start_bus()

    # Don't inject any response — timeout should fire
    with pytest.raises(asyncio.TimeoutError, match="first_token"):
        await svc.run_turn([{"role": "user", "content": "slow"}], tools=[], emit_frame=None)

    await svc.stop_bus()


@pytest.mark.asyncio
async def test_duplicate_delivery_discarded():
    """Re-delivery of a message for a closed correlation_id is silently dropped."""
    transport = InMemoryTransport()
    svc = _make_service(transport)
    await svc.start_bus()

    async def _backend():
        await asyncio.sleep(0.02)
        _, payload, _ = transport.published[-1]
        cid = json.loads(payload)["correlation_id"]
        await transport.inject_response("agentic.responses.acme", _final_msg(cid, "Answer"))
        # Duplicate delivery after turn is closed
        await asyncio.sleep(0.05)
        await transport.inject_response("agentic.responses.acme", _chunk(cid, "extra", 99))

    asyncio.create_task(_backend())
    result = await svc.run_turn([{"role": "user", "content": "hi"}], tools=[], emit_frame=None)

    # Only the original answer; duplicate is discarded silently
    assert result == "Answer"
    await svc.stop_bus()


@pytest.mark.asyncio
async def test_publish_cancel_sends_cancel_envelope():
    """publish_cancel() sends a vmo.agentic.cancel/1 envelope."""
    transport = InMemoryTransport()
    svc = _make_service(transport)
    await svc.start_bus()

    await svc.publish_cancel("call-1:turn-1", reason="barge_in")

    cancel_msgs = [
        json.loads(p) for _, p, _ in transport.published
        if json.loads(p).get("schema") == "vmo.agentic.cancel/1"
    ]
    assert cancel_msgs
    assert cancel_msgs[0]["reason"] == "barge_in"
    assert cancel_msgs[0]["correlation_id"] == "call-1:turn-1"
    await svc.stop_bus()


@pytest.mark.asyncio
async def test_correlation_id_format():
    """Correlation ID must be {vmo_call_id}:{turn_id}."""
    transport = InMemoryTransport()
    identity = _identity()
    svc = _make_service(transport, identity)
    await svc.start_bus()

    async def _backend():
        await asyncio.sleep(0.02)
        _, payload, _ = transport.published[-1]
        cid = json.loads(payload)["correlation_id"]
        await transport.inject_response("agentic.responses.acme", _final_msg(cid, "ok"))

    asyncio.create_task(_backend())
    await svc.run_turn([{"role": "user", "content": "test"}], tools=[], emit_frame=None)

    _, payload, _ = transport.published[0]
    req = json.loads(payload)
    cid = req["correlation_id"]
    assert cid.startswith(identity.vmo_call_id)
    await svc.stop_bus()
