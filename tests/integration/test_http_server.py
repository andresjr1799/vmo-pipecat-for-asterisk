"""
Integration tests — FastAPI HTTP server (Phase 10).

Tests /health/live, /health/ready, /metrics, /admin/reload, /admin/config/version,
/admin/calls/active using httpx AsyncClient with FastAPI TestClient pattern.
"""

from __future__ import annotations

import json
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import AsyncClient, ASGITransport

from vmo_pipecat.http.server import AppState, build_app


# ── Helpers ────────────────────────────────────────────────────────────────────

def _mock_state(
    config_valid: bool = True,
    ari_connected: bool = True,
    as_bound: bool = True,
    active_calls: int = 0,
    config_version: int = 3,
    admin_token: str = "secret",
    reload_ok: bool = True,
) -> AppState:
    config_store = MagicMock()
    config_store.is_valid = config_valid
    config_store.version = config_version
    config_store.info.return_value = {
        "config_version": config_version,
        "loaded_at_iso": "2026-05-08T00:00:00Z",
        "source_path": "/etc/vmo/tenants.yaml",
        "sha256": "abc123",
        "valid": config_valid,
    }

    registry = MagicMock()
    registry.active_count = active_calls
    registry.summary.return_value = []

    audiosocket = MagicMock()
    audiosocket.is_bound = as_bound

    pool = MagicMock()
    pool.configure_mock(**{"is_any_connected": ari_connected})

    event_bus = MagicMock()
    event_bus.emit = AsyncMock()

    reload_fn = AsyncMock(return_value=reload_ok)

    return AppState(
        config_store=config_store,
        registry=registry,
        audiosocket=audiosocket,
        pool=pool,
        event_bus=event_bus,
        admin_token=admin_token,
        reload_fn=reload_fn,
    )


@pytest.fixture
async def client():
    """httpx AsyncClient backed by the FastAPI ASGI app."""
    state = _mock_state()
    app = build_app(state)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c, state


async def _degraded_client():
    """Helper that creates an AsyncClient with ari_connected=False."""
    state = _mock_state(ari_connected=False)
    return build_app(state), state


# ═════════════════════════════════════════════════════════════════════════════
# /health/live
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_health_live_always_200(client):
    c, _ = client
    resp = await c.get("/health/live")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_health_live_200_even_when_config_invalid():
    state = _mock_state(config_valid=False)
    app = build_app(state)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.get("/health/live")
    assert resp.status_code == 200


# ═════════════════════════════════════════════════════════════════════════════
# /health/ready
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_health_ready_200_when_all_ok(client):
    c, _ = client
    resp = await c.get("/health/ready")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["config_valid"] is True
    assert body["ari_connected"] is True
    assert body["audiosocket_bound"] is True


@pytest.mark.asyncio
async def test_health_ready_503_when_ari_not_connected():
    app, state = await _degraded_client()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.get("/health/ready")
    assert resp.status_code == 503, f"Expected 503, got {resp.status_code}. Body: {resp.text}"
    body = resp.json()
    assert body["status"] == "degraded"
    assert body["ari_connected"] is False


@pytest.mark.asyncio
async def test_health_ready_503_when_config_invalid():
    state = _mock_state(config_valid=False, as_bound=True)
    app = build_app(state)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.get("/health/ready")
    assert resp.status_code == 503


@pytest.mark.asyncio
async def test_health_ready_includes_active_calls_count():
    state = _mock_state(active_calls=5)
    app = build_app(state)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.get("/health/ready")
    assert resp.json()["active_calls"] == 5


# ═════════════════════════════════════════════════════════════════════════════
# /metrics
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_metrics_returns_prometheus_text(client):
    c, _ = client
    resp = await c.get("/metrics")
    assert resp.status_code == 200
    assert "text/plain" in resp.headers["content-type"]
    assert "vmo_calls" in resp.text or "# HELP" in resp.text


# ═════════════════════════════════════════════════════════════════════════════
# /admin/reload
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_admin_reload_returns_200_with_valid_bearer(client):
    c, state = client
    resp = await c.post("/admin/reload",
                        headers={"Authorization": "Bearer secret"})
    assert resp.status_code == 200
    state.reload_fn.assert_awaited_once()


@pytest.mark.asyncio
async def test_admin_reload_returns_401_without_token(client):
    c, _ = client
    resp = await c.post("/admin/reload")
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_admin_reload_returns_401_with_wrong_token(client):
    c, _ = client
    resp = await c.post("/admin/reload",
                        headers={"Authorization": "Bearer wrong"})
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_admin_reload_returns_500_when_reload_fails():
    state = _mock_state(reload_ok=False, admin_token="tok")
    app = build_app(state)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.post("/admin/reload",
                            headers={"Authorization": "Bearer tok"})
    assert resp.status_code == 500


@pytest.mark.asyncio
async def test_admin_reload_200_when_no_token_configured():
    """When admin_token is empty, any request is accepted."""
    state = _mock_state(admin_token="")
    app = build_app(state)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.post("/admin/reload")
    assert resp.status_code == 200


# ═════════════════════════════════════════════════════════════════════════════
# /admin/config/version
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_admin_config_version_returns_info(client):
    c, _ = client
    resp = await c.get("/admin/config/version")
    assert resp.status_code == 200
    body = resp.json()
    assert body["config_version"] == 3
    assert "sha256" in body


# ═════════════════════════════════════════════════════════════════════════════
# /admin/calls/active
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_admin_calls_active_returns_count_and_list(client):
    c, _ = client
    resp = await c.get("/admin/calls/active")
    assert resp.status_code == 200
    body = resp.json()
    assert "active_count" in body
    assert "calls" in body
    assert isinstance(body["calls"], list)


@pytest.mark.asyncio
async def test_admin_calls_active_count_matches_registry():
    state = _mock_state(active_calls=7)
    app = build_app(state)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.get("/admin/calls/active")
    assert resp.json()["active_count"] == 7
