# Changelog

All notable changes to **VMO-PipeCat-For-Asterisk** are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased — V1 feature-complete, Phases 0–11]

## [0.1.0] — Phase 0 — Bootstrap

### Added
- `vmo_pipecat/` namespace created; all new code lives here.
- **Inherited from `vmo-engine`** (zero logic changes, only import path updates):
  - `vmo_pipecat/audio/audiosocket_server.py` — AsyncSocket TLV server.
  - `vmo_pipecat/audio/resampler.py` — μ-law ↔ PCM16 + resample helpers.
  - `vmo_pipecat/ari/{client,pool,events}.py` — multi-Asterisk ARI client with exponential reconnect.
  - `vmo_pipecat/observability/log_setup.py` — structlog JSON config (renamed from `logging_config.py`; service name updated to `vmo-pipecat`).
- Prometheus metric names updated to `vmo_audiosocket_*` namespace (§8.5).
- `pyproject.toml` with all V1 dependencies (`pipecat-ai[deepgram,elevenlabs,openai,silero,aws]`, FastAPI, structlog, prometheus-client, watchdog, …).
- `Dockerfile` — `python:3.11-slim`, exposes 8090 and 15000, healthcheck via `/health/live`.
- `docker-compose.yml` — `vmo_pipecat` + `vmo_asterisk_1` + `vmo_asterisk_2` with healthcheck dependency.
- `.env.example` — template with all required env vars.
- `Makefile` — `up`, `down`, `logs`, `test`, `lint`, `e2e-*` targets.
- `README.md` — quick start, port table, phase status.
- `tenants.example.yaml` — complete multi-tenant example (acme + globex, all provider kinds).
- `vmo_pipecat/__main__.py` + `vmo_pipecat/runtime.py` — async entrypoint skeleton with SIGTERM graceful shutdown.
- Empty package stubs: `config/`, `transport/`, `call/`, `pipelines/`, `providers/`, `bus/`, `tenancy/`, `actions/`, `events/`, `http/`.
- `tests/unit/` and `tests/integration/` with empty suites (pytest green).

### Removed
- `vmo_engine/` directory (contents migrated to `vmo_pipecat/`).

### Changed
- `asterisk/conf-templates/extensions.conf.tmpl` — updated to the minimal dialplan from §11.2 (all calls → Stasis, auxiliary `from-vmo-transfer` context).
