"""
FastAPI HTTP server for vmo-pipecat-for-asterisk (Phase 10).

Endpoints (§3.1):
  GET  /health/live        → 200 always (liveness probe)
  GET  /health/ready       → 200 iff config valid + ARI connected + AudioSocket bound
  GET  /metrics            → Prometheus text format (scraped by Prometheus / Grafana)
  POST /admin/reload       → Bearer auth, force hot-reload of tenants.yaml
  GET  /admin/config/version → config snapshot info
  GET  /admin/calls/active → active call summary

The app is constructed by build_app() and run via uvicorn in runtime.py.
All state is injected via AppState (no globals in this module).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable, Optional

from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse
from starlette.responses import Response as StarletteResponse


# ── AppState (injected by runtime) ─────────────────────────────────────────────

@dataclass
class AppState:
    """Holds references to the live singletons passed from runtime.main()."""
    config_store: Any
    registry: Any
    audiosocket: Any
    pool: Any
    event_bus: Any
    admin_token: str
    reload_fn: Callable[[], Awaitable[bool]]


# ── App factory ────────────────────────────────────────────────────────────────

def build_app(state: AppState) -> FastAPI:
    """Build the FastAPI application with all endpoints wired to live state."""

    app = FastAPI(
        title="VMO-PipeCat-For-Asterisk",
        version="1.0",
        docs_url=None,   # disable Swagger UI in production
        redoc_url=None,
    )
    app.state.vmo = state

    # ── /health ────────────────────────────────────────────────────────────────

    @app.get("/health/live", tags=["health"])
    async def health_live() -> dict:
        """Liveness probe — no deep checks."""
        return {"status": "ok"}

    @app.get("/health/ready", tags=["health"])
    async def health_ready(response: Response) -> dict:
        """Readiness probe: config valid + ARI ≥1 node + AudioSocket bound."""
        s: AppState = app.state.vmo
        config_ok = bool(s.config_store.is_valid)
        ari_ok = bool(s.pool.is_any_connected)
        as_ok = bool(s.audiosocket.is_bound)
        all_ok = config_ok and ari_ok and as_ok

        if not all_ok:
            response.status_code = 503

        return {
            "status": "ok" if all_ok else "degraded",
            "config_valid": config_ok,
            "config_version": s.config_store.version,
            "ari_connected": ari_ok,
            "audiosocket_bound": as_ok,
            "active_calls": s.registry.active_count,
        }

    # ── /admin ─────────────────────────────────────────────────────────────────

    def _check_bearer(authorization: Optional[str]) -> None:
        s: AppState = app.state.vmo
        expected = f"Bearer {s.admin_token}" if s.admin_token else None
        if expected and authorization != expected:
            raise HTTPException(status_code=401, detail="Unauthorized")

    @app.post("/admin/reload", tags=["admin"])
    async def admin_reload(
        authorization: Optional[str] = Header(default=None),
    ) -> JSONResponse:
        """Force hot-reload of tenants.yaml (requires Bearer token)."""
        _check_bearer(authorization)
        s: AppState = app.state.vmo
        ok = await s.reload_fn()
        code = 200 if ok else 500
        return JSONResponse(content=s.config_store.info(), status_code=code)

    @app.get("/admin/config/version", tags=["admin"])
    async def admin_config_version() -> dict:
        s: AppState = app.state.vmo
        return s.config_store.info()

    @app.get("/admin/calls/active", tags=["admin"])
    async def admin_calls_active() -> dict:
        s: AppState = app.state.vmo
        return {
            "active_count": s.registry.active_count,
            "calls": s.registry.summary(),
        }

    return app
