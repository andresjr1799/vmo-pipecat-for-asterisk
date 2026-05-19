"""
from_asterisk.py — translates Asterisk ARI / AudioSocket events into PipeCat frames.

Frames are injected into the active PipelineTask via task.queue_frame().

Supported translations (§4.2):
  ChannelDtmfReceived  → DTMFFrame(digit, channel_id)
  AudioSocket TYPE_DTMF → DTMFFrame(digit, channel_id)  [same path, via CallRouter]
  ChannelTalkingStarted  → UserStartedSpeakingFrame  (only when vad=asterisk_talk_detect)
  ChannelTalkingFinished → UserStoppedSpeakingFrame   (only when vad=asterisk_talk_detect)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ── PipeCat base frame (graceful fallback) ─────────────────────────────────────
try:
    from pipecat.frames.frames import Frame as _PipecatFrame
    from pipecat.frames.frames import UserStartedSpeakingFrame, UserStoppedSpeakingFrame
    _PIPECAT = True
except ImportError:
    _PIPECAT = False

    @dataclass
    class _PipecatFrame:  # type: ignore[no-redef]
        """Minimal Frame stub when pipecat-ai is not installed."""

    @dataclass
    class UserStartedSpeakingFrame(_PipecatFrame):  # type: ignore[no-redef]
        pass

    @dataclass
    class UserStoppedSpeakingFrame(_PipecatFrame):  # type: ignore[no-redef]
        pass


# ── DTMFFrame ──────────────────────────────────────────────────────────────────

@dataclass
class DTMFFrame(_PipecatFrame):
    """Carries a DTMF digit received from Asterisk into the PipeCat pipeline.

    If the LLM context has a registered tool named `on_dtmf` (or a processor
    that consumes DTMFFrame), the pipeline will deliver it there.

    When PipeCat is installed, this is a proper PipeCat Frame and can be
    routed by the pipeline's frame-dispatch mechanism.
    """
    digit: str = ""
    channel_id: str = ""


# ── Factory helpers ────────────────────────────────────────────────────────────

def make_user_started_speaking() -> Any:
    """Return a UserStartedSpeakingFrame (PipeCat native or stub)."""
    return UserStartedSpeakingFrame()


def make_user_stopped_speaking() -> Any:
    """Return a UserStoppedSpeakingFrame (PipeCat native or stub)."""
    return UserStoppedSpeakingFrame()
