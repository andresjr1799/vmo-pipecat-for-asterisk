"""
PipeCat pipeline stubs — used when pipecat-ai is not installed.

The stub PipelineRunner.run() blocks until task.cancel() is called, mimicking
how a real PipeCat runner stays alive processing frames until the pipeline ends.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional


class _StubPipelineTask:
    """Minimal PipelineTask substitute that blocks run() until cancelled."""

    def __init__(self, pipeline: Any, params: Any = None) -> None:
        self.pipeline = pipeline
        self._done: asyncio.Event = asyncio.Event()

    async def queue_frame(self, frame: Any) -> None:
        pass

    async def cancel(self) -> None:
        self._done.set()


class _StubPipelineRunner:
    """Minimal PipelineRunner substitute that blocks until task is cancelled."""

    async def run(self, task: Any) -> None:
        done_event = getattr(task, "_done", None)
        if done_event is not None:
            await done_event.wait()
        # else: immediate return (task has no blocking mechanism)


class _StubPipeline:
    def __init__(self, processors: Any) -> None:
        self.processors = processors


class _StubPipelineParams:
    def __init__(self, **kw: Any) -> None:
        for k, v in kw.items():
            setattr(self, k, v)
