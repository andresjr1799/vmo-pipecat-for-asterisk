"""
build_modular_pipeline — pipeline PipeCat modular para una llamada (PipeCat 1.1.0+).

Estructura:
  transport.input() → STT → user_aggregator → LLM → TTS → transport.output() → assist_aggregator

PipeCat 1.1.0 cambios vs 0.x:
  - LLMContext + LLMContextMessage: pipecat.processors.aggregators.llm_context
  - LLMContextAggregatorPair: pipecat.processors.aggregators.llm_response_universal
  - SileroVADAnalyzer NO es un FrameProcessor → VAD nativo del STT (Deepgram endpointing)
  - OpenAI usa OPENAI_API_KEY env var (no api_key en constructor en 1.1.0)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Tuple, Optional

if TYPE_CHECKING:
    from ..tenancy.resolver import SessionConfig
    from ..transport.asterisk_transport import AsteriskAudioSocketTransport
    from ..actions.asterisk_actions import AsteriskActions

from ..providers.registry import build_provider
from ..actions.tool_schemas import schemas_for_tools
from ..observability.log_setup import get_logger
from ..observability.pipeline_metrics import MetricsFrameProcessor
from ..call.metrics import CallMetrics

logger = get_logger(__name__)


class _NoopProcessor:
    """Pass-through when metrics are disabled."""
    def __init__(self): pass
    def link(self, n): pass
    async def process_frame(self, frame, direction): pass


# ── PipeCat 1.1.0 imports ──────────────────────────────────────────────────────
try:
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.runner import PipelineRunner
    from pipecat.pipeline.task import PipelineTask, PipelineParams
    from pipecat.processors.aggregators.llm_context import LLMContext, LLMContextMessage, NOT_GIVEN
    from pipecat.processors.aggregators.llm_response_universal import (
        LLMContextAggregatorPair,
        LLMUserAggregatorParams,
    )
    from pipecat.audio.vad.silero import SileroVADAnalyzer
    from pipecat.audio.vad.vad_analyzer import VADParams
    _PIPECAT = True
except ImportError:
    _PIPECAT = False
    NOT_GIVEN = None  # stub sentinel
    from ._stubs import (
        _StubPipeline as Pipeline,              # type: ignore[assignment]
        _StubPipelineRunner as PipelineRunner,  # type: ignore[assignment]
        _StubPipelineTask as PipelineTask,      # type: ignore[assignment]
        _StubPipelineParams as PipelineParams,  # type: ignore[assignment]
    )

    class LLMContext:  # type: ignore[no-redef]
        def __init__(self, messages=None, tools=None): self._messages = messages or []

    class LLMContextMessage:  # type: ignore[no-redef]
        def __init__(self, role, content): self.role = role; self.content = content

    class _AggStub:
        def __init__(self): pass
        def link(self, n): pass

    class LLMContextAggregatorPair:  # type: ignore[no-redef]
        def __init__(self, ctx, **kw): pass
        def user(self): return _AggStub()
        def assistant(self): return _AggStub()

    class LLMUserAggregatorParams:  # type: ignore[no-redef]
        def __init__(self, **kw): pass

    class SileroVADAnalyzer:  # type: ignore[no-redef]
        def __init__(self, **kw): pass

    class VADParams:  # type: ignore[no-redef]
        def __init__(self, **kw): pass


def build_modular_pipeline(
    session: "SessionConfig",
    transport: "AsteriskAudioSocketTransport",
    actions: "AsteriskActions",
    call_metrics: Optional["CallMetrics"] = None,
) -> Tuple[Any, Any]:
    """Build a modular STT + LLM + TTS PipeCat 1.1.0 pipeline.

    Returns (runner, task).
    """
    from ..config.models import ModularPipelineCfg
    pipeline_cfg: ModularPipelineCfg = session.pipeline  # type: ignore[assignment]
    audio = session.audio_profile

    # ── Providers ─────────────────────────────────────────────────────────────
    stt = build_provider(session.providers[pipeline_cfg.stt], audio)
    tts = build_provider(session.providers[pipeline_cfg.tts], audio)

    llm_resolved = session.providers[pipeline_cfg.llm]
    if llm_resolved.kind == "agentic_bus":
        llm = build_provider(
            llm_resolved, audio,
            identity=session.identity,
            session_config=session,
        )
    else:
        llm = build_provider(llm_resolved, audio)

    # ── LLM context (PipeCat 1.1.0 API) ───────────────────────────────────────
    # LLMContextMessage es un Union — se usan dicts planos estilo OpenAI
    tool_names = list(session.context.tools or [])
    tool_schemas = schemas_for_tools(tool_names) or None

    context = LLMContext(
        messages=[{"role": "system", "content": session.context.prompt or ""}],
        tools=tool_schemas if tool_schemas else NOT_GIVEN,
    )

    # ── VAD (desde YAML, no hardcodeado) ──────────────────────────────────────
    vad_kind = getattr(pipeline_cfg, "vad", "silero") or "silero"
    vad_analyzer = None

    if vad_kind == "silero":
        from ..config.models import VADCfg, VADOverridesCfg, OverridesCfg
        # Resolver: pipeline.vad_params > tenant overrides (que ya incluyen defaults)
        ov = getattr(session.overrides, "vad", None) or VADOverridesCfg()
        vp = getattr(pipeline_cfg, "vad_params", None) or VADCfg()

        start_secs = float(ov.speech_threshold_ms or 150) / 1000.0
        stop_secs  = float(ov.silence_threshold_ms or 200) / 1000.0

        if vp.start_secs is not None:
            start_secs = float(vp.start_secs)
        if vp.stop_secs is not None:
            stop_secs = float(vp.stop_secs)

        vad_analyzer = SileroVADAnalyzer(
            params=VADParams(stop_secs=stop_secs, start_secs=start_secs)
        )
        logger.info("VAD: silero", stop_secs=stop_secs, start_secs=start_secs)

    elif vad_kind == "asterisk_talk_detect":
        logger.info("VAD: asterisk_talk_detect (via ARI ChannelTalking events)")

    elif vad_kind == "none":
        logger.info("VAD: none (transcription-based turn detection)")

    user_params = LLMUserAggregatorParams(vad_analyzer=vad_analyzer)
    context_pair = LLMContextAggregatorPair(context, user_params=user_params)

    # ── Tool registration ──────────────────────────────────────────────────────
    action_map = {
        "transfer_call": actions.transfer_call,
        "hangup_call": actions.hangup_call,
        "play_audio_file": actions.play_audio_file,
        "send_dtmf": actions.send_dtmf,
    }
    for name in tool_names:
        if name in action_map and hasattr(llm, "register_function"):
            llm.register_function(name, action_map[name])

    # ── Pipeline (sin SileroVAD — no es FrameProcessor en 1.1.0) ─────────────
    # El endpointing lo maneja el STT service (Deepgram tiene VAD nativo)

    # Observability processors
    if call_metrics:
        call_metrics.set_providers(pipeline_cfg.stt, pipeline_cfg.llm, pipeline_cfg.tts)
        stt_metrics = MetricsFrameProcessor(call_metrics, "stt")
        llm_metrics = MetricsFrameProcessor(call_metrics, "llm")
        tts_metrics = MetricsFrameProcessor(call_metrics, "tts")
    else:
        stt_metrics = llm_metrics = tts_metrics = None

    processors: list[Any] = [
        transport.input(),
        stt,
        stt_metrics if stt_metrics else _NoopProcessor(),
        context_pair.user(),
        llm,
        llm_metrics if llm_metrics else _NoopProcessor(),
        tts,
        tts_metrics if tts_metrics else _NoopProcessor(),
        transport.output(),
        context_pair.assistant(),
    ]

    pipeline = Pipeline(processors)
    params = PipelineParams(
        audio_in_sample_rate=audio.in_rate,
        audio_out_sample_rate=audio.out_rate,
    )
    task = PipelineTask(pipeline, params=params)
    runner = PipelineRunner()

    logger.info(
        "Modular pipeline built (PipeCat 1.1.0)",
        stt=pipeline_cfg.stt,
        llm=pipeline_cfg.llm,
        tts=pipeline_cfg.tts,
        tools=tool_names,
    )
    return runner, task
