"""
CallLifecycle — handles Asterisk ARI events that drive call setup/teardown.

Implements §4.1 of the architecture document.

IMPORTANTE: Las variables de canal (tenant_id, tenant_name, etc.) se leen
usando GET /channels/{id}/variable via ARI REST API, NO desde channelvars
del evento StasisStart. Esto replica el patrón del VMO Engine original y
funciona sin necesidad de configurar chanvars en ari.conf.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

import structlog

from .controller import CallController
from .identity import CallIdentity
from ..ari.events import (
    STASIS_START,
    STASIS_END,
    CHANNEL_DESTROYED,
    CHANNEL_DTMF_RECEIVED,
    NODE_ID_KEY,
)
from ..observability.log_setup import get_logger
from ..observability.otel import get_tracer
from ..transport.asterisk_transport import AsteriskAudioSocketTransport

if TYPE_CHECKING:
    from ..ari.pool import ARIPool
    from ..audio.audiosocket_server import AudioSocketServer
    from ..call.router import CallRouter
    from ..call.registry import CallRegistry
    from ..events.bus import EventBus
    from ..tenancy.resolver import TenantResolver

logger = get_logger(__name__)


async def _get_channel_var(ari, channel_id: str, var_name: str) -> str:
    """Lee una variable de canal via ARI REST API GET.

    Patrón idéntico al VMO Engine original — no depende de channelvars
    del evento StasisStart (que puede llegar vacío sin chanvars en ari.conf).
    """
    try:
        resp = await ari.send_command(
            "GET",
            f"channels/{channel_id}/variable",
            params={"variable": var_name},
            tolerate_statuses=[404],
        )
        if isinstance(resp, dict):
            return (resp.get("value") or "").strip()
    except Exception as exc:
        logger.debug("Failed to read channel variable", channel_id=channel_id, var=var_name, error=str(exc))
    return ""


class CallLifecycle:
    """Wires ARI events into call setup/teardown logic."""

    def __init__(
        self,
        pool: "ARIPool",
        audiosocket: "AudioSocketServer",
        router: "CallRouter",
        registry: "CallRegistry",
        resolver: "TenantResolver",
        event_bus: "EventBus",
        stasis_app: str = "vmo-pipecat-app",
    ) -> None:
        self._pool = pool
        self._audiosocket = audiosocket
        self._router = router
        self._registry = registry
        self._resolver = resolver
        self._event_bus = event_bus
        self._stasis_app = stasis_app
        # Transient map: audio_uuid → (controller, bridge_id)
        self._pending_audio: dict[str, tuple[CallController, str]] = {}

    # ------------------------------------------------------------------
    # ARI event handlers
    # ------------------------------------------------------------------

    async def _on_stasis_start(self, event: dict) -> None:
        try:
            await self._handle_stasis_start(event)
        except Exception as exc:
            logger.error("StasisStart handler error", error=str(exc), exc_info=True)

    async def _on_stasis_end(self, event: dict) -> None:
        try:
            await self._handle_stasis_end(event)
        except Exception as exc:
            logger.error("StasisEnd handler error", error=str(exc), exc_info=True)

    async def _on_channel_destroyed(self, event: dict) -> None:
        channel_id = event.get("channel", {}).get("id", "")
        ctrl = self._registry.get_by_channel(channel_id)
        if ctrl:
            await ctrl.shutdown(reason="channel_destroyed")

    async def _on_dtmf_received(self, event: dict) -> None:
        channel_id = event.get("channel", {}).get("id", "")
        digit = event.get("digit", "")
        if digit:
            ctrl = self._registry.get_by_channel(channel_id)
            if ctrl:
                await ctrl.on_dtmf(digit)

    async def _on_channel_talking_started(self, event: dict) -> None:
        try:
            channel_id = event.get("channel", {}).get("id", "")
            ctrl = self._registry.get_by_channel(channel_id)
            if ctrl and hasattr(ctrl, "on_talking_started"):
                await ctrl.on_talking_started()
        except Exception as exc:
            logger.error("ChannelTalkingStarted handler error", error=str(exc))

    async def _on_channel_talking_finished(self, event: dict) -> None:
        try:
            channel_id = event.get("channel", {}).get("id", "")
            ctrl = self._registry.get_by_channel(channel_id)
            if ctrl and hasattr(ctrl, "on_talking_finished"):
                await ctrl.on_talking_finished()
        except Exception as exc:
            logger.error("ChannelTalkingFinished handler error", error=str(exc))

    # ------------------------------------------------------------------
    # StasisStart dispatch
    # ------------------------------------------------------------------

    async def _handle_stasis_start(self, event: dict) -> None:
        channel = event.get("channel", {})
        channel_id: str = channel.get("id", "")
        channel_name: str = channel.get("name", "")
        node_id: str = event.get(NODE_ID_KEY, "")

        # Detectar canal AudioSocket: por nombre de canal
        is_audiosocket = channel_name.startswith("AudioSocket/")

        if is_audiosocket:
            await self._on_audiosocket_channel(event, channel_id, node_id)
        else:
            await self._on_caller_channel(event, channel_id, node_id)

    # ------------------------------------------------------------------
    # AudioSocket channel entra a Stasis → leer UUID via API y añadir al bridge
    # ------------------------------------------------------------------

    async def _on_audiosocket_channel(
        self, event: dict, channel_id: str, node_id: str
    ) -> None:
        # Leer AUDIOSOCKET_UUID via GET API — no depender de channelvars
        try:
            ari = self._pool.client_for(node_id)
        except KeyError:
            logger.warning("No ARI client for node on audiosocket channel", node_id=node_id)
            return

        audio_uuid = await _get_channel_var(ari, channel_id, "AUDIOSOCKET_UUID")

        if not audio_uuid:
            # Fallback: intentar extraer del nombre del canal (AudioSocket/<host>:<port>/<uuid>)
            parts = channel_name_to_uuid(event.get("channel", {}).get("name", ""))
            audio_uuid = parts

        if not audio_uuid:
            logger.warning("AudioSocket channel: no AUDIOSOCKET_UUID", channel_id=channel_id)
            return

        entry = self._pending_audio.pop(audio_uuid, None)
        if entry is None:
            logger.warning("No pending call for audio_uuid", audio_uuid=audio_uuid)
            return

        controller, bridge_id = entry
        controller._audio_channel_id = channel_id
        try:
            # Responder el canal AudioSocket → estado Up → bridge entrega audio bidireccional
            await ari.answer_channel(channel_id)
            await ari.add_channel_to_bridge(bridge_id, channel_id)
            logger.info(
                "AudioSocket channel added to bridge",
                audio_uuid=audio_uuid,
                bridge_id=bridge_id,
                **controller.identity.asdict(),
            )
        except Exception as exc:
            logger.error("Failed to add AudioSocket channel to bridge",
                         audio_uuid=audio_uuid, error=str(exc))

    # ------------------------------------------------------------------
    # Caller channel entra a Stasis — setup completo
    # ------------------------------------------------------------------

    async def _on_caller_channel(
        self, event: dict, caller_channel_id: str, node_id: str
    ) -> None:
        # Obtener ARI client
        try:
            ari = self._pool.client_for(node_id)
        except KeyError:
            logger.error("No ARI client for node", node_id=node_id)
            return

        # [5] Leer variables via GET API (patrón VMO Engine original)
        tenant_id   = await _get_channel_var(ari, caller_channel_id, "tenant_id")
        tenant_name = await _get_channel_var(ari, caller_channel_id, "tenant_name")
        call_id_sbc = await _get_channel_var(ari, caller_channel_id, "call_id_sbc")
        caller_id   = await _get_channel_var(ari, caller_channel_id, "caller_id")
        did_var     = await _get_channel_var(ari, caller_channel_id, "did")

        # DID: preferir variable explícita, luego args de Stasis, luego EXTEN
        args = event.get("args") or []
        did = did_var or (args[0] if args else "")

        # Si tenant_id vacío → el fallback del resolver se encarga
        vmo_call_id = str(uuid.uuid4())
        identity = CallIdentity(
            vmo_call_id=vmo_call_id,
            asterisk_channel_id=caller_channel_id,
            call_id_sbc=call_id_sbc,
            caller_id=caller_id,
            tenant_id=tenant_id,
            tenant_name=tenant_name,
            node_id=node_id,
            did=did,
        )

        structlog.contextvars.bind_contextvars(**identity.asdict())
        logger.info("Caller channel in Stasis",
                    tenant_id=tenant_id, tenant_name=tenant_name,
                    call_id_sbc=call_id_sbc, did=did)

        # ── OTel: span padre vmo.call (sin context.attach, solo start_span) ──
        call_span = get_tracer().start_span(
            "vmo.call",
            attributes={
                "telecom.sbc.call_id": call_id_sbc,
                "telecom.asterisk.channel_id": caller_channel_id,
                "telecom.caller.id": caller_id,
                "business.vmo_call_id": vmo_call_id,
                "business.tenant_id": tenant_id,
                "business.did": did,
            },
        )
        # Inject trace_id/span_id into structlog contextvars (evita context.attach)
        span_ctx = call_span.get_span_context()
        if span_ctx.is_valid:
            structlog.contextvars.bind_contextvars(
                trace_id=format(span_ctx.trace_id, "032x"),
                span_id=format(span_ctx.span_id, "016x"),
            )

        await self._event_bus.emit(
            "vmo.call.started",
            identity,
            did=did,
            arrived_at_iso=datetime.now(tz=timezone.utc).isoformat(),
        )

        # Resolve session config (frozen)
        try:
            session_config = self._resolver.resolve(tenant_id, did, identity=identity)
        except Exception as exc:
            logger.error("Failed to resolve session config", error=str(exc))
            try:
                await ari.hangup_channel(caller_channel_id)
            except Exception:
                pass
            return

        await self._event_bus.emit(
            "vmo.call.config_resolved",
            identity,
            pipeline_kind=session_config.pipeline.kind,
        )

        # [6] ARI: answer + bridge + add caller + originate AudioSocket
        await ari.answer_channel(caller_channel_id)

        # Bridge mixing puro — sin proxy_media ni dtmf_events.
        # proxy_media intenta RTP directo pero AudioSocket es TCP → incompatible.
        bridge_id = await ari.create_bridge("mixing")
        if not bridge_id:
            logger.error("Failed to create bridge")
            await ari.hangup_channel(caller_channel_id)
            return

        await ari.add_channel_to_bridge(bridge_id, caller_channel_id)

        audio_uuid = str(uuid.uuid4())
        advertise_host = os.getenv("VMO_AUDIOSOCKET_ADVERTISE_HOST", "127.0.0.1")
        port = self._audiosocket.port
        codec = session_config.audio_profile.codec
        endpoint = f"AudioSocket/{advertise_host}:{port}/{audio_uuid}/c({codec})"

        # Build transport + controller ANTES del originate para evitar race condition.
        # El TCP UUID handshake puede llegar en el mismo ms que el originate.
        transport = AsteriskAudioSocketTransport(self._audiosocket, session_config.audio_profile)
        controller = CallController(
            identity=identity,
            session_config=session_config,
            bridge_id=bridge_id,
            pool=self._pool,
            audiosocket=self._audiosocket,
            transport=transport,
            router=self._router,
            registry=self._registry,
            event_bus=self._event_bus,
            call_span=call_span,
        )

        # Registrar ANTES del originate — el UUID handshake TCP puede llegar
        # en el mismo instante en que Asterisk origina el canal.
        await self._router.register_pending(audio_uuid, controller)
        self._router.register_by_vmo_call_id(vmo_call_id, controller)
        self._router.register_by_channel(caller_channel_id, controller)
        await self._registry.add(controller)
        self._pending_audio[audio_uuid] = (controller, bridge_id)

        originate_result = await ari.send_command(
            "POST",
            "channels",
            params={
                "endpoint": endpoint,
                "app": self._stasis_app,
                "timeout": "30",
                "channelVars": {"AUDIOSOCKET_UUID": audio_uuid},
            },
        )
        if not originate_result or originate_result.get("status", 200) >= 400:
            logger.error("Failed to originate AudioSocket channel",
                         endpoint=endpoint, result=originate_result)
            self._router.remove(controller)
            await self._registry.remove(controller)
            self._pending_audio.pop(audio_uuid, None)
            await ari.hangup_channel(caller_channel_id)
            return

        asyncio.create_task(
            controller.start(),
            name=f"ctrl-{vmo_call_id[:8]}",
        )

        logger.info("Call setup complete",
                    audio_uuid=audio_uuid,
                    bridge_id=bridge_id,
                    endpoint=endpoint)

    # ------------------------------------------------------------------
    # StasisEnd → shutdown
    # ------------------------------------------------------------------

    async def _handle_stasis_end(self, event: dict) -> None:
        channel = event.get("channel", {})
        channel_id: str = channel.get("id", "")
        channel_name: str = channel.get("name", "")

        if channel_name.startswith("AudioSocket/"):
            return

        ctrl = self._registry.get_by_channel(channel_id)
        if ctrl:
            await ctrl.shutdown(reason="stasis_end")


def channel_name_to_uuid(channel_name: str) -> str:
    """Extrae el UUID del nombre del canal AudioSocket como fallback.

    Formato del canal: AudioSocket/<host>:<port>/<uuid>[-<suffix>]
    """
    if not channel_name.startswith("AudioSocket/"):
        return ""
    try:
        parts = channel_name.split("/")
        # parts[0]="AudioSocket", parts[1]="host:port", parts[2]="uuid[-suffix]"
        if len(parts) >= 3:
            uuid_part = parts[2].split("-")[0:5]   # UUID tiene 5 grupos con guion
            return "-".join(uuid_part)
    except Exception:
        pass
    return ""
