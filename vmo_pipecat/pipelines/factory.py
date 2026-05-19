"""
PipelineFactory — dispatches to the correct pipeline builder by pipeline.kind.

Usage in CallController.start():
    runner, task = PipelineFactory.build(session_config, transport, actions)
    await emit_greeting(task, session_config)
    await runner.run(task)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple, Any, Optional

if TYPE_CHECKING:
    from ..tenancy.resolver import SessionConfig
    from ..transport.asterisk_transport import AsteriskAudioSocketTransport
    from ..actions.asterisk_actions import AsteriskActions
    from ..call.metrics import CallMetrics

from ..observability.log_setup import get_logger

logger = get_logger(__name__)


class PipelineFactory:
    """Stateless factory — all state lives in the per-call session_config."""

    @staticmethod
    def build(
        session: "SessionConfig",
        transport: "AsteriskAudioSocketTransport",
        actions: "AsteriskActions",
        call_metrics: Optional["CallMetrics"] = None,
    ) -> Tuple[Any, Any]:
        """Build (PipelineRunner, PipelineTask) for this call's session config.

        Dispatches by pipeline.kind:
          "modular"    → STT + LLM + TTS with SileroVAD (Phase 4+)
          "full_agent" → single full-agent WebSocket (Phase 5+)

        Raises:
            ValueError: if pipeline.kind is unknown.
        """
        kind = session.pipeline.kind

        if kind == "modular":
            from .modular import build_modular_pipeline
            return build_modular_pipeline(session, transport, actions, call_metrics)

        if kind == "full_agent":
            from .full_agent import build_full_agent_pipeline
            return build_full_agent_pipeline(session, transport, actions)

        raise ValueError(f"Unknown pipeline kind: '{kind}'")
