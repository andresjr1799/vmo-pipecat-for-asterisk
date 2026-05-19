"""
AgenticBusLLMService — PipeCat LLMService that delegates each turn to an
external backend agentic system via the message bus (§9.5).

In a modular pipeline, this service occupies the LLM slot:
  transport.input() → VAD → STT → user_aggregator → AgenticBusLLMService → TTS

For each turn:
  1. Receives LLM context frames from the aggregator.
  2. Publishes a `vmo.agentic.request/1` envelope to `outbound.destination`.
  3. Subscribes to `inbound.destination` and routes responses by correlation_id.
  4. Emits TextFrame(s) for each chunk/final → TTS processes them.
  5. On tool_call: invokes the registered AsteriskActions callback.
  6. On barge-in (BotInterruptionFrame): publishes cancel + stops consuming.

PipeCat import note: when pipecat-ai is not installed, falls back to a
lightweight stub so the module is importable for testing the bus logic.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from ..bus.correlation import CorrelationRouter
from ..bus.envelope import (
    AgenticCancelEnvelope,
    AgenticToolResultEnvelope,
    build_request,
    parse_response,
)
from ..observability.log_setup import get_logger

if TYPE_CHECKING:
    from ..bus import AgenticBusTransport
    from ..call.identity import CallIdentity
    from ..tenancy.resolver import SessionConfig

logger = get_logger(__name__)

# ── PipeCat imports with graceful fallbacks ────────────────────────────────────

try:
    from pipecat.services.openai.llm import OpenAILLMService as _LLMServiceBase
    from pipecat.frames.frames import (
        TextFrame,
        LLMMessagesFrame,
        LLMFullResponseStartFrame,
        LLMFullResponseEndFrame,
    )
    _PIPECAT = True
except ImportError:
    _PIPECAT = False

    class _LLMServiceBase:  # type: ignore[no-redef]
        def __init__(self, **kw): self._functions: dict = {}
        def register_function(self, name: str, cb: Any): self._functions[name] = cb

    class TextFrame:  # type: ignore[no-redef]
        def __init__(self, text: str): self.text = text

    class LLMMessagesFrame:  # type: ignore[no-redef]
        def __init__(self, messages): self.messages = messages

    class LLMFullResponseStartFrame:  # type: ignore[no-redef]
        pass

    class LLMFullResponseEndFrame:  # type: ignore[no-redef]
        pass


# ── AgenticBusLLMService ───────────────────────────────────────────────────────

class AgenticBusLLMService(_LLMServiceBase):
    """LLM service that delegates turns to an external backend via message bus.

    One instance per call. The transport is shared per-tenant (one connection
    per `connection.servers` + `auth_signature` combination, cached externally).
    """

    def __init__(
        self,
        transport: "AgenticBusTransport",
        outbound_destination: str,
        inbound_destination: str,
        inbound_group: Optional[str],
        identity: "CallIdentity",
        session_config: "SessionConfig",
        provider_params: Dict[str, Any],
    ) -> None:
        if _PIPECAT:
            # Pass minimal OpenAI-compatible params so PipeCat initializes correctly
            super().__init__(api_key="agentic_bus", model="agentic_bus")
        else:
            super().__init__()

        self._transport = transport
        self._outbound = outbound_destination
        self._inbound = inbound_destination
        self._inbound_group = inbound_group
        self._identity = identity
        self._session_config = session_config
        self._params = provider_params
        self._functions: Dict[str, Callable] = {}
        self._correlation_router = CorrelationRouter()
        self._turn_counter = 0
        self._subscription = None

        # Config from provider params
        self._streaming = provider_params.get("streaming", "tokens")
        self._first_token_timeout_ms = int(provider_params.get("first_token_timeout_ms", 4000))
        self._request_timeout_ms = int(provider_params.get("request_timeout_ms", 15000))
        self._tool_call_handling = provider_params.get("tool_call_handling", "bus")
        self._thread_id_strategy = provider_params.get("thread_id_strategy", "vmo_call_id")
        self._assistant_hint = provider_params.get("assistant_hint", "")
        self._greeting_strategy = provider_params.get("greeting_strategy", "local")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start_bus(self) -> None:
        """Subscribe to inbound destination. Must be called once at call start."""
        async def _on_inbound(payload: bytes, headers: dict) -> None:
            try:
                msg = parse_response(payload)
                await self._correlation_router.deliver(msg.model_dump())
            except Exception as exc:
                logger.warning("Failed to parse inbound bus message", error=str(exc))

        self._subscription = await self._transport.subscribe(
            self._inbound,
            self._inbound_group,
            _on_inbound,
        )
        logger.info("AgenticBusLLMService: subscribed to inbound", destination=self._inbound)

    async def stop_bus(self) -> None:
        if self._subscription:
            await self._subscription.unsubscribe()
            self._subscription = None

    # ------------------------------------------------------------------
    # Core: run one LLM turn
    # ------------------------------------------------------------------

    async def run_turn(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        emit_frame: Optional[Callable] = None,
    ) -> str:
        """Execute one LLM turn: publish request → collect response.

        Args:
            messages: OpenAI-format messages (system + conversation history).
            tools: OpenAI-format tool schemas to advertise to the backend.
            emit_frame: Optional async callable to forward TextFrame to the pipeline.

        Returns:
            Concatenated text response from the backend.
        """
        self._turn_counter += 1
        turn_id = str(uuid.uuid4())[:8]
        correlation_id = f"{self._identity.vmo_call_id}:{turn_id}"

        # Thread ID strategy
        if self._thread_id_strategy == "vmo_call_id":
            thread_id = self._identity.vmo_call_id
        elif self._thread_id_strategy == "per_turn":
            thread_id = correlation_id
        else:
            thread_id = self._identity.vmo_call_id

        # Register correlation slot
        response_queue = self._correlation_router.register(correlation_id)

        # Build and publish request envelope
        envelope = build_request(
            correlation_id=correlation_id,
            identity_dict=self._identity.asdict(),
            reply_to=self._inbound,
            messages=messages,
            tools=tools,
            session={
                "thread_id": thread_id,
                "assistant_hint": self._assistant_hint,
            },
            preferences={
                "streaming": self._streaming,
                "language": "es",
            },
            deadlines={
                "first_token_ms": self._first_token_timeout_ms,
                "complete_response_ms": self._request_timeout_ms,
            },
        )

        try:
            await self._transport.publish(
                self._outbound,
                envelope.to_bytes(),
                {"Content-Type": "application/json"},
            )
        except Exception as exc:
            self._correlation_router.close(correlation_id)
            logger.error("Bus publish failed", error=str(exc), correlation_id=correlation_id)
            raise

        logger.info(
            "AgenticBus: request published",
            correlation_id=correlation_id,
            outbound=self._outbound,
        )

        # Collect response
        return await self._collect_response(
            correlation_id, response_queue, emit_frame
        )

    async def _collect_response(
        self,
        correlation_id: str,
        queue: asyncio.Queue,
        emit_frame: Optional[Callable],
    ) -> str:
        """Wait for and process inbound response messages for this turn."""
        full_text = []
        first_token_deadline = self._first_token_timeout_ms / 1000.0
        total_deadline = self._request_timeout_ms / 1000.0
        got_first_token = False
        t0 = time.monotonic()

        try:
            while True:
                elapsed = time.monotonic() - t0
                remaining = total_deadline - elapsed

                # First-token timeout
                if not got_first_token:
                    timeout = min(first_token_deadline - elapsed, remaining)
                else:
                    timeout = remaining

                if timeout <= 0:
                    phase = "first_token" if not got_first_token else "complete"
                    logger.error(
                        "AgenticBus: timeout",
                        correlation_id=correlation_id,
                        phase=phase,
                    )
                    raise asyncio.TimeoutError(f"Agentic bus timeout ({phase})")

                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=timeout)
                except asyncio.TimeoutError:
                    phase = "first_token" if not got_first_token else "complete"
                    raise asyncio.TimeoutError(f"Agentic bus timeout ({phase})")

                if msg is None:
                    # Sentinel — correlation was cancelled externally
                    break

                msg_type = msg.get("type", "")

                if msg_type == "chunk":
                    got_first_token = True
                    text = msg.get("text", "")
                    if text:
                        full_text.append(text)
                        if emit_frame:
                            await emit_frame(TextFrame(text=text))

                elif msg_type == "tool_call":
                    got_first_token = True
                    tc = msg.get("tool_call", {})
                    await self._handle_tool_call(tc, correlation_id)

                elif msg_type == "final":
                    got_first_token = True
                    text = msg.get("text", "")
                    if text:
                        full_text.append(text)
                        if emit_frame:
                            await emit_frame(TextFrame(text=text))
                    break

                elif msg_type == "end":
                    break

                elif msg_type == "error":
                    err = msg.get("error", {})
                    logger.error(
                        "AgenticBus: error from backend",
                        correlation_id=correlation_id,
                        error=err,
                    )
                    break

        finally:
            self._correlation_router.close(correlation_id)

        return "".join(full_text)

    async def _handle_tool_call(
        self,
        tool_call: Dict[str, Any],
        correlation_id: str,
    ) -> None:
        """Invoke the registered callback for a tool_call from the backend."""
        name = tool_call.get("name", "")
        tool_call_id = tool_call.get("id", "")
        arguments = tool_call.get("arguments", {})

        cb = self._functions.get(name)
        if cb is None:
            logger.warning("AgenticBus: unknown tool_call name", name=name)
            return

        result_holder: List[Any] = []

        async def result_callback(result: Any) -> None:
            result_holder.append(result)

        try:
            await cb(name, tool_call_id, arguments, None, None, result_callback)
        except Exception as exc:
            logger.error("AgenticBus: tool_call callback error", name=name, error=str(exc))
            result_holder.append({"status": "error", "error": str(exc)})

        # Publish tool_result back to the backend (if bus mode)
        if self._tool_call_handling == "bus" and result_holder:
            result_env = AgenticToolResultEnvelope(
                correlation_id=correlation_id,
                tool_call_id=tool_call_id,
                result=result_holder[0] if isinstance(result_holder[0], dict) else {"value": result_holder[0]},
            )
            try:
                await self._transport.publish(
                    self._outbound,
                    result_env.to_bytes(),
                    {"Content-Type": "application/json"},
                )
            except Exception as exc:
                logger.warning("AgenticBus: failed to publish tool_result", error=str(exc))

    # ------------------------------------------------------------------
    # function registration (compatible with PipeCat LLMService.register_function)
    # ------------------------------------------------------------------

    def register_function(self, name: str, callback: Any) -> None:
        self._functions[name] = callback

    async def publish_cancel(self, correlation_id: str, reason: str = "barge_in") -> None:
        """Publish cancel envelope and stop consuming the current turn."""
        cancel_env = AgenticCancelEnvelope(
            correlation_id=correlation_id,
            reason=reason,
        )
        try:
            await self._transport.publish(
                self._outbound,
                cancel_env.to_bytes(),
                {"Content-Type": "application/json"},
            )
        except Exception as exc:
            logger.warning("AgenticBus: failed to publish cancel", error=str(exc))
        finally:
            self._correlation_router.close(correlation_id)


# ── Provider builder ───────────────────────────────────────────────────────────

def build_service(
    resolved: Any,
    audio_profile: Any,
    *,
    identity: Any = None,
    session_config: Any = None,
) -> "AgenticBusLLMService":
    """Build an AgenticBusLLMService from a resolved AgenticBusProviderCfg.

    identity and session_config are required at call time and passed by
    build_modular_pipeline() when it detects kind=agentic_bus.
    """
    from ..bus import make_bus_transport

    params = dict(getattr(resolved, "params", {}))
    transport_kind = getattr(resolved, "transport", "nats")
    connection_cfg = dict(getattr(resolved, "connection", {}) or {})
    auth_cfg = dict(getattr(resolved, "auth", {}) or {})
    outbound_cfg = getattr(resolved, "outbound", None)
    inbound_cfg = getattr(resolved, "inbound", None)

    if identity is None or session_config is None:
        raise NotImplementedError(
            "AgenticBusLLMService requires identity and session_config. "
            "Use build_modular_pipeline() which passes these automatically."
        )

    transport = make_bus_transport(
        transport_kind,
        {"connection": connection_cfg, "auth": auth_cfg},
    )

    outbound_dest = getattr(outbound_cfg, "destination", "") if outbound_cfg else ""
    inbound_dest = getattr(inbound_cfg, "destination", "") if inbound_cfg else ""
    inbound_group = getattr(inbound_cfg, "group", None) if inbound_cfg else None

    return AgenticBusLLMService(
        transport=transport,
        outbound_destination=outbound_dest,
        inbound_destination=inbound_dest,
        inbound_group=inbound_group,
        identity=identity,
        session_config=session_config,
        provider_params=params,
    )
