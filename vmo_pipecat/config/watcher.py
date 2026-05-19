"""
ConfigWatcher — hot-reload of tenants.yaml via watchdog.

Watches the file's parent directory for changes and debounces (500 ms)
before triggering a reload callback in the asyncio event loop.
Uses call_soon_threadsafe to safely hand off from the watchdog thread.
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Awaitable, Callable, Optional

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from ..observability.log_setup import get_logger

logger = get_logger(__name__)


class _Handler(FileSystemEventHandler):
    """Watchdog handler that forwards relevant events to the asyncio loop."""

    def __init__(
        self,
        target_path: str,
        trigger: Callable[[], None],
    ) -> None:
        super().__init__()
        self._target = str(Path(target_path).resolve())
        self._trigger = trigger

    def _is_target(self, path: str) -> bool:
        return str(Path(path).resolve()) == self._target

    def on_modified(self, event: FileSystemEvent) -> None:
        if not event.is_directory and self._is_target(event.src_path):
            self._trigger()

    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory and self._is_target(event.src_path):
            self._trigger()

    def on_moved(self, event: FileSystemEvent) -> None:
        # Editors (vim, emacs) often write to a temp file then rename
        if not event.is_directory and self._is_target(getattr(event, "dest_path", "")):
            self._trigger()


class ConfigWatcher:
    """Watches a YAML file and calls `on_change()` (async) after debounce."""

    def __init__(
        self,
        path: str,
        on_change: Callable[[], Awaitable[None]],
        debounce_ms: int = 500,
    ) -> None:
        self._path = str(Path(path).resolve())
        self._on_change = on_change
        self._debounce_s = debounce_ms / 1000.0
        self._observer: Optional[Observer] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._debounce_handle: Optional[asyncio.TimerHandle] = None
        self._lock = threading.Lock()

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        handler = _Handler(self._path, self._on_file_event_from_thread)
        watch_dir = str(Path(self._path).parent)
        self._observer = Observer()
        self._observer.schedule(handler, path=watch_dir, recursive=False)
        self._observer.start()
        logger.info("ConfigWatcher started", path=self._path, debounce_ms=int(self._debounce_s * 1000))

    async def stop(self) -> None:
        if self._observer:
            self._observer.stop()
            self._observer.join()
            self._observer = None
        logger.info("ConfigWatcher stopped")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_file_event_from_thread(self) -> None:
        """Called from the watchdog OS thread; bounces to asyncio loop."""
        if self._loop is None:
            return
        self._loop.call_soon_threadsafe(self._schedule_reload)

    def _schedule_reload(self) -> None:
        """Runs in the asyncio event loop; cancels previous timer, sets new one."""
        with self._lock:
            if self._debounce_handle is not None:
                self._debounce_handle.cancel()
            self._debounce_handle = self._loop.call_later(
                self._debounce_s,
                lambda: asyncio.ensure_future(self._on_change()),
            )
