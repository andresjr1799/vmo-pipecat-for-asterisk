FROM python:3.11-slim

LABEL org.opencontainers.image.title="VMO-PipeCat-For-Asterisk" \
      org.opencontainers.image.description="Voice AI engine for Asterisk using PipeCat"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libffi-dev \
        libssl-dev \
        curl \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Actualizar pip y setuptools antes de cualquier instalación
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Instalar dependencias de producción directamente (sin editable install)
# pipecat-ai[silero] incluye PyTorch — la layer más pesada, se cachea aquí
RUN pip install --no-cache-dir \
    "pipecat-ai[deepgram,elevenlabs,openai,silero,aws]>=0.0.46"

# Resto de dependencias del proyecto
RUN pip install --no-cache-dir \
    "aiohttp>=3.9" \
    "websockets>=12.0" \
    "pydantic>=2.0" \
    "pyyaml>=6.0" \
    "structlog>=24.0" \
    "opentelemetry-api>=1.25" \
    "opentelemetry-sdk>=1.25" \
    "opentelemetry-exporter-otlp-proto-grpc>=1.25" \
    "opentelemetry-instrumentation-logging>=0.46b0" \
    "watchdog>=4.0" \
    "fastapi>=0.111" \
    "uvicorn[standard]>=0.29" \
    "webrtcvad>=2.0.10" \
    "numpy>=1.26"

# Copiar código fuente
COPY vmo_pipecat/ ./vmo_pipecat/

EXPOSE 8090 15000

HEALTHCHECK --interval=5s --timeout=3s --start-period=60s --retries=15 \
    CMD curl -fsS http://localhost:15000/health/live || exit 1

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

CMD ["python", "-m", "vmo_pipecat"]
