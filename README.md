# vmo-pipecat-for-asterisk

Voice AI engine for Asterisk PBX using **PipeCat** as the voice orchestration framework.

## STT + LLM + TTS — Pipelines funcionales


## Quick Start

```bash
# 1. Copy and fill credentials
cp .env.example .env

# 2. Create tenants.yaml
cp tenants.example.yaml tenants.yaml

# 3. Start
docker compose up -d

# 4. Check health
curl http://localhost:15000/health/ready
```

## .env variables required

```env
DEEPGRAM_API_KEY=       # STT + TTS + Full-Agent
OPENAI_API_KEY=          # LLM
ELEVENLABS_API_KEY=      # STT + TTS + Full-Agent
EL_VOICE_ID=             # ElevenLabs TTS voice
EL_AGENT_ID=             # ElevenLabs Full-Agent
AWS_KEY= / AWS_SECRET=   # AWS Transcribe STT (opcional)

# Asterisk
AST1_ARI_USERNAME= / AST1_ARI_PASSWORD=
AST1_HOST=               # Asterisk IP
HOST_IP_NIC= / LOCAL_NET= / PASS_EXTENSIONS_TEST=
RTP_START= / RTP_END=
```

## Audio Flow

```
Asterisk ←──AudioSocket TCP──→ vmo-pipecat ←──WebSocket──→ Deepgram / ElevenLabs / AWS
                                        ↕
                                   OpenAI (LLM)
```

- AudioSocket: `/c(slin)` 8kHz PCM16 input → resample 16kHz TTS output
- Deepgram STT: linear16 8kHz
- ElevenLabs TTS: WebSocket pcm_16000 → resample 8kHz → AudioSocket

## Architecture

See [arquitectura-vmo-pipecat-for-asterisk.md](arquitectura-vmo-pipecat-for-asterisk.md)

## Ports

| Port | Protocol | Purpose |
|------|----------|---------|
| 8090 | TCP | AudioSocket |
| 15000 | HTTP | `/health`, `/metrics`, `/admin` |
| 5060 | UDP/TCP | Asterisk SIP |
| 8088 | HTTP | Asterisk ARI |

## Hot Reload

Edit `tenants.yaml` → watcher detects change in 500ms → atomic swap. Active calls unaffected.
