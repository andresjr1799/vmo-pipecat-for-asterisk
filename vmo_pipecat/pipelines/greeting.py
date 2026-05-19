"""
emit_greeting — sends the context greeting to the TTS before the first user turn.

How it's delivered depends on pipeline.kind (§9.4):
- modular/agentic_bus: TTSSpeakFrame queued directly into PipelineTask.
- full_agent: greeting is passed to the provider builder; do NOT call here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..tenancy.resolver import SessionConfig

try:
    from pipecat.frames.frames import TTSSpeakFrame
    _PIPECAT = True
except ImportError:
    _PIPECAT = False

    class TTSSpeakFrame:  # type: ignore[no-redef]
        def __init__(self, text: str):
            self.text = text


async def emit_greeting(task: Any, session: "SessionConfig") -> None:
    """Queue the greeting TextFrame into the PipelineTask (modular pipelines).

    For full_agent pipelines, the greeting is delivered via the provider's
    native mechanism during build_service() and must NOT be queued here.
    """
    from ..config.models import FullAgentPipelineCfg
    if isinstance(session.pipeline, FullAgentPipelineCfg):
        return  # handled by provider builder

    text = (session.context.greeting or "").strip()
    if not text:
        return

    if _PIPECAT and hasattr(task, "queue_frame"):
        await task.queue_frame(TTSSpeakFrame(text))
    # If PipeCat stubs are in use, greeting is logged but not actually spoken
