"""
VMO-PipeCat-For-Asterisk — async runtime supervisor (Phase 10).

Wires all components and manages their full lifecycle:
  ConfigStore + ConfigWatcher   → hot-reload (§7)
  ARIPool (instrumented)        → multi-Asterisk ARI WebSocket
  AudioSocketServer             → TLV TCP listener (singleton)
  CallRouter + CallRegistry     → AudioSocket → CallController dispatch
  CallLifecycle                 → ARI event → call setup/teardown
  FastAPI + uvicorn             → /health, /metrics, /admin (§3.1)
  Graceful shutdown             → 30 s drain → forced cleanup (§10)
"""

from __future__ import annotations

import asyncio
import os
import signal
from pathlib import Path

import pydantic
import uvicorn

from .ari.client import ARIClient
from .ari.events import (
    STASIS_START, STASIS_END, CHANNEL_DESTROYED,
    CHANNEL_DTMF_RECEIVED, CHANNEL_TALKING_STARTED, CHANNEL_TALKING_FINISHED,
)
from .ari.pool import ARIPool
from .audio.audiosocket_server import AudioSocketServer
from .call.lifecycle import CallLifecycle
from .call.registry import CallRegistry
from .call.router import CallRouter
from .config.loader import load_tenants_config
from .config.models import TenantsConfig
from .config.store import ConfigStore
from .config.watcher import ConfigWatcher
from .events.bus import LoggingEventBus
from .http.server import AppState, build_app
from .observability.log_setup import configure_logging, get_logger
from .observability.otel_metrics import (
    record_config_reload,
    record_ari_reconnect,
    set_ari_node_connected,
)
from .tenancy.resolver import TenantResolver

logger = get_logger(__name__)

_shutdown_event: asyncio.Event
_config_store = ConfigStore()

# ── Instrumented ARIClient ──────────────────────────────────────────────────────

class _InstrumentedARIClient(ARIClient):
    """ARIClient subclass that increments vmo_ari_reconnect_total on reconnect."""

    async def _mark_disconnected_and_backoff(self, *args, **kwargs) -> bool:
        result = await super()._mark_disconnected_and_backoff(*args, **kwargs)
        if result:   # True = about to attempt reconnect
            record_ari_reconnect(self.node_id)
        return result


# ── Config helpers ─────────────────────────────────────────────────────────────

async def _do_reload(config_path: str, event_bus=None) -> bool:
    try:
        cfg, sha = load_tenants_config(config_path)
        _config_store.swap(cfg, source_path=config_path, sha256=sha)
        logger.info(
            "vmo.system.config.reloaded",
            config_version=_config_store.version,
            sha256=sha,
        )
        record_config_reload("ok")
        if event_bus:
            await event_bus.emit(
                "vmo.system.config.reloaded",
                config_version=_config_store.version,
                source_path=config_path,
                sha256=sha,
            )
        return True
    except FileNotFoundError:
        logger.error("Config file not found", path=config_path)
        _config_store.mark_invalid()
        record_config_reload("invalid")
        if event_bus:
            await event_bus.emit("vmo.system.config.invalid", errors=["File not found"])
        return False
    except pydantic.ValidationError as exc:
        errors = exc.errors()
        logger.error("vmo.system.config.invalid", errors=errors)
        _config_store.mark_invalid()
        record_config_reload("invalid")
        if event_bus:
            await event_bus.emit("vmo.system.config.invalid", errors=errors)
        return False
    except Exception as exc:
        logger.error("Config reload failed", error=str(exc), exc_info=True)
        _config_store.mark_invalid()
        record_config_reload("invalid")
        return False


# ── ARIPool builder ─────────────────────────────────────────────────────────────

def _build_pool(cfg: TenantsConfig) -> ARIPool:
    pool = ARIPool()
    for node in cfg.asterisk_nodes:
        scheme = getattr(node, "ari_scheme", "http") or "http"
        base_url = f"{scheme}://{node.host}:{node.ari_port}/ari"
        client = _InstrumentedARIClient(
            username=node.ari_username,
            password=node.ari_password,
            base_url=base_url,
            app_name=node.ari_app,
            node_id=node.id,
        )
        pool.add_node(node.id, client)
    return pool


# ── Graceful shutdown ───────────────────────────────────────────────────────────

async def _drain_calls(registry: CallRegistry, timeout_s: float = 30.0) -> None:
    """Wait for active calls to finish; force-shutdown remaining after timeout."""
    active = registry.active_count
    if active == 0:
        logger.info("shutdown.drained", active_calls=0)
        return

    logger.info("shutdown.started", active_calls=active)
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout_s

    while registry.active_count > 0:
        if loop.time() >= deadline:
            remaining = registry.active_count
            logger.warning("shutdown.forced", remaining_calls=remaining)
            for ctrl in list(registry.all_controllers()):
                await ctrl.shutdown(reason="forced_shutdown")
            await asyncio.sleep(0.1)
            break
        await asyncio.sleep(0.5)

    logger.info("shutdown.drained", remaining_calls=registry.active_count)


# ── HTTP (FastAPI + uvicorn) ────────────────────────────────────────────────────

async def _run_http(
    state: AppState,
    host: str,
    port: int,
) -> None:
    app = build_app(state)
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(config)
    logger.info("HTTP server starting", host=host, port=port)
    try:
        await server.serve()
    finally:
        logger.info("HTTP server stopped")


# ── Main ─────────────────────────────────────────────────────────────────────────

async def main() -> None:
    global _shutdown_event
    _shutdown_event = asyncio.Event()

    configure_logging(log_level=os.getenv("VMO_LOG_LEVEL", "INFO"))

    # ── OpenTelemetry SDK ─────────────────────────────────────────────────────
    from .observability.otel import init_otel
    from .observability.otel_metrics import init_metrics
    init_otel()
    init_metrics()

    logger.info(
        "VMO-PipeCat-For-Asterisk starting",
        version=os.getenv("VMO_VERSION", "dev"),
        pid=os.getpid(),
    )

    config_path = os.getenv("VMO_CONFIG_PATH", "tenants.yaml")
    admin_token = os.getenv("VMO_ADMIN_TOKEN", "")
    stasis_app = os.getenv("STASIS_APP", "vmo-pipecat-app")
    http_host = os.getenv("VMO_HTTP_HOST", "0.0.0.0")
    http_port = int(os.getenv("VMO_HTTP_PORT", "15000"))
    drain_timeout = float(os.getenv("VMO_DRAIN_TIMEOUT_S", "30"))

    loop = asyncio.get_running_loop()

    # Core singletons
    resolver = TenantResolver(_config_store)
    router = CallRouter()
    registry = CallRegistry()
    event_bus = LoggingEventBus()

    def _handle_signal(sig: signal.Signals) -> None:
        logger.info("Shutdown signal received", signal=sig.name)
        _shutdown_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _handle_signal, sig)

    # Config: initial load
    if Path(config_path).exists():
        await _do_reload(config_path, event_bus=event_bus)
    else:
        logger.warning("Config file not found at startup", path=config_path)

    # One-time startup read — permitted exception to the ConfigStore.current rule.
    # This happens BEFORE any calls arrive, not during an active call.
    cfg = _config_store.current

    # AudioSocket TCP listener
    audiosocket = AudioSocketServer(
        host=os.getenv("VMO_AUDIOSOCKET_HOST", "0.0.0.0"),
        port=int(os.getenv("VMO_AUDIOSOCKET_PORT", "8090")),
        on_uuid=lambda conn_id, uid: router.bind_uuid(uid, conn_id),
        on_audio=router.dispatch_audio,
        on_disconnect=router.dispatch_disconnect,
        on_dtmf=router.dispatch_dtmf,
    )
    await audiosocket.start()

    # ARIPool (instrumented)
    pool = _build_pool(cfg) if cfg else ARIPool()

    # Emit initial ARI node state
    if cfg:
        for node in cfg.asterisk_nodes:
            set_ari_node_connected(node.id, False)
            await event_bus.emit(
                "vmo.system.ari.node_state",
                node_id=node.id,
                state="connecting",
                attempt=0,
            )

    # CallLifecycle
    lifecycle = CallLifecycle(
        pool=pool, audiosocket=audiosocket, router=router,
        registry=registry, resolver=resolver, event_bus=event_bus,
        stasis_app=stasis_app,
    )
    pool.on_event(STASIS_START, lifecycle._on_stasis_start)
    pool.on_event(STASIS_END, lifecycle._on_stasis_end)
    pool.on_event(CHANNEL_DESTROYED, lifecycle._on_channel_destroyed)
    pool.on_event(CHANNEL_DTMF_RECEIVED, lifecycle._on_dtmf_received)
    pool.on_event(CHANNEL_TALKING_STARTED, lifecycle._on_channel_talking_started)
    pool.on_event(CHANNEL_TALKING_FINISHED, lifecycle._on_channel_talking_finished)

    # Hot-reload watcher
    watcher: ConfigWatcher | None = None
    if Path(config_path).exists():
        watcher = ConfigWatcher(
            config_path,
            on_change=lambda: _do_reload(config_path, event_bus=event_bus),
        )
        await watcher.start()

    # FastAPI / uvicorn HTTP server
    http_state = AppState(
        config_store=_config_store,
        registry=registry,
        audiosocket=audiosocket,
        pool=pool,
        event_bus=event_bus,
        admin_token=admin_token,
        reload_fn=lambda: _do_reload(config_path, event_bus=event_bus),
    )

    tasks = [
        asyncio.create_task(
            _run_http(http_state, http_host, http_port),
            name="http-server",
        ),
        asyncio.create_task(
            pool.start_all_listening(),
            name="ari-pool",
        ),
    ]

    logger.info("Runtime supervisor ready — all components started")
    await _shutdown_event.wait()

    # ── Graceful shutdown ───────────────────────────────────────────────────────
    logger.info("Graceful shutdown initiated")

    # 1. Stop accepting new calls
    if watcher:
        await watcher.stop()

    # 2. Drain active calls (30 s timeout)
    await _drain_calls(registry, timeout_s=drain_timeout)

    # 3. Disconnect infrastructure
    await pool.stop_all()
    await audiosocket.stop()
    await event_bus.close()

    # 4. Cancel remaining tasks
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    logger.info("VMO-PipeCat-For-Asterisk stopped cleanly")
