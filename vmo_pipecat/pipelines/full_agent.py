"""
build_full_agent_pipeline — single-WS full-agent pipeline (Phase 5).

Full-agent providers (Deepgram Voice Agent, ElevenLabs Conversational) handle
STT + LLM + TTS natively in one WebSocket. The pipeline is much simpler than
the modular variant:
  transport.input() → agent_service → transport.output()

Context fields (prompt, greeting) are passed to the provider builder and
translated to native provider parameters (§9.4.2).

VAD and interruption are handled natively by the full-agent provider;
SileroVAD is NOT added to this pipeline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple, Any

if TYPE_CHECKING:
    from ..tenancy.resolver import SessionConfig
    from ..transport.asterisk_transport import AsteriskAudioSocketTransport
    from ..actions.asterisk_actions import AsteriskActions

from ..providers.registry import build_provider
from ..observability.log_setup import get_logger

logger = get_logger(__name__)

try:
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.runner import PipelineRunner
    from pipecat.pipeline.task import PipelineTask, PipelineParams
    _PIPECAT = True
except ImportError:
    _PIPECAT = False
    from ._stubs import (
        _StubPipeline as Pipeline,              # type: ignore[assignment]
        _StubPipelineRunner as PipelineRunner,  # type: ignore[assignment]
        _StubPipelineTask as PipelineTask,      # type: ignore[assignment]
        _StubPipelineParams as PipelineParams,  # type: ignore[assignment]
    )


def build_full_agent_pipeline(
    session: "SessionConfig",
    transport: "AsteriskAudioSocketTransport",
    actions: "AsteriskActions",
) -> Tuple[PipelineRunner, PipelineTask]:
    """Build a full-agent pipeline (Deepgram Voice Agent / ElevenLabs Conv)."""
    from ..config.models import FullAgentPipelineCfg
    pipeline_cfg: FullAgentPipelineCfg = session.pipeline  # type: ignore[assignment]
    audio = session.audio_profile

    agent = build_provider(
        session.providers[pipeline_cfg.provider],
        audio,
        # Pass context fields so provider builder can inject prompt/greeting
        context_prompt=session.context.prompt or "",
        context_greeting=session.context.greeting or "",
    )

    # Register LLM tool functions on the full-agent service
    action_map = {
        "transfer_call": actions.transfer_call,
        "hangup_call": actions.hangup_call,
        "play_audio_file": actions.play_audio_file,
        "send_dtmf": actions.send_dtmf,
    }
    for name in (session.context.tools or []):
        if name in action_map and hasattr(agent, "register_function"):
            agent.register_function(name, action_map[name])

    pipeline = Pipeline([transport.input(), agent, transport.output()])
    params = PipelineParams(
        audio_in_sample_rate=audio.in_rate,
        audio_out_sample_rate=audio.out_rate,
    )
    task = PipelineTask(pipeline, params=params)
    runner = PipelineRunner()

    logger.info(
        "Full-agent pipeline built",
        provider=pipeline_cfg.provider,
        tools=list(session.context.tools or []),
    )
    return runner, task
