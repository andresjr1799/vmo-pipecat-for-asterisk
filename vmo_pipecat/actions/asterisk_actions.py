"""
AsteriskActions — LLM function-call callbacks that execute ARI operations.

One instance per call. Callbacks follow PipeCat's function-calling convention:
    async def handler(function_name, tool_call_id, args, llm, context, result_callback)

ARI calls always use pool.client_for(identity.node_id) — never a global client.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Awaitable

from ..observability.log_setup import get_logger
from ..observability.otel_metrics import record_transfer, record_tool_call

if TYPE_CHECKING:
    from ..ari.pool import ARIPool
    from ..call.identity import CallIdentity
    from ..events.bus import EventBus

logger = get_logger(__name__)


class AsteriskActions:
    """Adapter: LLM tool-call → ARI command + EventBus emit.

    Instantiated once per call; identity is baked in so callbacks don't
    need to carry extra context.
    """

    def __init__(
        self,
        pool: "ARIPool",
        identity: "CallIdentity",
        transfer_context: str,
        event_bus: "EventBus",
    ) -> None:
        self._pool = pool
        self._identity = identity
        self._transfer_context = transfer_context
        self._event_bus = event_bus
        self._transfer_done = False   # idempotency guard

    # ------------------------------------------------------------------
    # PipeCat function-call handlers
    # ------------------------------------------------------------------

    async def transfer_call(
        self,
        function_name: str,
        tool_call_id: str,
        args: dict,
        llm: Any,
        context: Any,
        result_callback: Callable[..., Awaitable],
    ) -> None:
        """Transfer the call via ARI continue_in_dialplan."""
        if self._transfer_done:
            await result_callback({"status": "already_transferred"})
            return
        self._transfer_done = True

        target = str(args.get("target", self._identity.did))
        reason = str(args.get("reason", "transfer"))
        logger.info("AsteriskActions: transfer_call", target=target, reason=reason)

        try:
            ari = self._pool.client_for(self._identity.node_id)
            ok = await ari.continue_in_dialplan(
                self._identity.asterisk_channel_id,
                context=self._transfer_context,
                extension=target,
                priority=1,
            )
            await self._event_bus.emit(
                "vmo.call.transfer.requested",
                self._identity,
                target=target,
                reason=reason,
            )
            result = "ok" if ok else "failed"
            if ok:
                await self._event_bus.emit("vmo.call.transfer.done", self._identity, target=target)
            else:
                await self._event_bus.emit(
                    "vmo.call.transfer.failed", self._identity, target=target, error="ARI rejected"
                )
            record_transfer(self._identity.tenant_id, result)
            await result_callback({"status": result, "target": target})
        except Exception as exc:
            logger.error("transfer_call failed", error=str(exc), exc_info=True)
            await self._event_bus.emit(
                "vmo.call.transfer.failed", self._identity, target=target, error=str(exc)
            )
            record_transfer(self._identity.tenant_id, "failed")
            await result_callback({"status": "error", "error": str(exc)})

    async def hangup_call(
        self,
        function_name: str,
        tool_call_id: str,
        args: dict,
        llm: Any,
        context: Any,
        result_callback: Callable[..., Awaitable],
    ) -> None:
        """Hang up the caller channel via ARI."""
        reason = str(args.get("reason", "llm_initiated"))
        logger.info("AsteriskActions: hangup_call", reason=reason)
        try:
            ari = self._pool.client_for(self._identity.node_id)
            await ari.hangup_channel(self._identity.asterisk_channel_id)
            await self._event_bus.emit(
                "vmo.call.tool_call", self._identity, name="hangup_call", outcome="ok"
            )
            record_tool_call(self._identity.tenant_id, "hangup_call", "ok")
            await result_callback({"status": "ok"})
        except Exception as exc:
            logger.error("hangup_call failed", error=str(exc), exc_info=True)
            record_tool_call(self._identity.tenant_id, "hangup_call", "error")
            await result_callback({"status": "error", "error": str(exc)})

    async def play_audio_file(
        self,
        function_name: str,
        tool_call_id: str,
        args: dict,
        llm: Any,
        context: Any,
        result_callback: Callable[..., Awaitable],
    ) -> None:
        """Play a media file via ARI play_media."""
        uri = str(args.get("uri", ""))
        if not uri:
            await result_callback({"status": "error", "error": "missing uri"})
            return
        logger.info("AsteriskActions: play_audio_file", uri=uri)
        try:
            ari = self._pool.client_for(self._identity.node_id)
            await ari.play_media(self._identity.asterisk_channel_id, uri)
            await self._event_bus.emit(
                "vmo.call.tool_call", self._identity, name="play_audio_file", outcome="ok", uri=uri
            )
            record_tool_call(self._identity.tenant_id, "play_audio_file", "ok")
            await result_callback({"status": "ok", "uri": uri})
        except Exception as exc:
            logger.error("play_audio_file failed", error=str(exc), exc_info=True)
            record_tool_call(self._identity.tenant_id, "play_audio_file", "error")
            await result_callback({"status": "error", "error": str(exc)})

    async def send_dtmf(
        self,
        function_name: str,
        tool_call_id: str,
        args: dict,
        llm: Any,
        context: Any,
        result_callback: Callable[..., Awaitable],
    ) -> None:
        """Send DTMF digits via ARI."""
        digits = str(args.get("digits", ""))
        if not digits:
            await result_callback({"status": "error", "error": "missing digits"})
            return
        logger.info("AsteriskActions: send_dtmf", digits=digits)
        try:
            ari = self._pool.client_for(self._identity.node_id)
            await ari.send_command(
                "POST",
                f"channels/{self._identity.asterisk_channel_id}/dtmf",
                params={"dtmf": digits},
            )
            await self._event_bus.emit(
                "vmo.call.tool_call", self._identity, name="send_dtmf", outcome="ok", digits=digits
            )
            record_tool_call(self._identity.tenant_id, "send_dtmf", "ok")
            await result_callback({"status": "ok", "digits": digits})
        except Exception as exc:
            logger.error("send_dtmf failed", error=str(exc), exc_info=True)
            record_tool_call(self._identity.tenant_id, "send_dtmf", "error")
            await result_callback({"status": "error", "error": str(exc)})
