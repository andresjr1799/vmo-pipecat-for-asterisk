"""Async AudioSocket server for Asterisk AudioSocket integrations.

Implements the TLV protocol: UUID handshake, audio frames, DTMF, TERMINATE/ERROR.
Singleton listener; one CallController per connection via CallRouter.
"""

from __future__ import annotations

import asyncio
import contextlib
import socket
import uuid
from typing import Awaitable, Callable, Dict, Optional

from vmo_pipecat.observability.log_setup import get_logger
from vmo_pipecat.observability.otel_metrics import (
    audiosocket_conn_inc,
    audiosocket_conn_dec,
    audiosocket_rx_bytes_inc,
    audiosocket_tx_bytes_inc,
)

logger = get_logger(__name__)

# AudioSocket TLV message types
TYPE_TERMINATE = 0x00
TYPE_UUID = 0x01
TYPE_DTMF = 0x03
TYPE_AUDIO = 0x10       # slin 8kHz PCM (forzado con /c(slin) en endpoint)
TYPE_AUDIO_16 = 0x18    # slin16 16kHz PCM — Asterisk 20 sin codec forzado
TYPE_ERROR = 0xFF


class AudioSocketServer:
    """Async AudioSocket server with TLV parsing and callback hooks."""

    def __init__(
        self,
        host: str,
        port: int,
        *,
        on_uuid: Callable[[str, str], Awaitable[bool]],
        on_audio: Callable[[str, bytes], Awaitable[None]],
        on_disconnect: Optional[Callable[[str], Awaitable[None]]] = None,
        on_dtmf: Optional[Callable[[str, str], Awaitable[None]]] = None,
    ) -> None:
        self.host = host
        self.port = port
        self._on_uuid = on_uuid
        self._on_audio = on_audio
        self._on_disconnect = on_disconnect
        self._on_dtmf = on_dtmf

        self._server: Optional[asyncio.base_events.Server] = None
        self._connection_tasks: Dict[str, asyncio.Task] = {}
        self._writers: Dict[str, asyncio.StreamWriter] = {}
        self._conn_to_uuid: Dict[str, str] = {}
        self._lock = asyncio.Lock()
        self._first_audio_logged: Dict[str, bool] = {}
        self._closed_logged: set[str] = set()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def start(self) -> None:
        if self._server:
            logger.warning("AudioSocket server already running")
            return

        self._server = await asyncio.start_server(
            self._handle_client,
            host=self.host,
            port=self.port,
        )

        sockets = self._server.sockets or []
        if sockets:
            self.port = sockets[0].getsockname()[1]

        logger.info("AudioSocket server listening", host=self.host, port=self.port)

    async def stop(self) -> None:
        async with self._lock:
            tasks = list(self._connection_tasks.values())
            writers = list(self._writers.values())
            self._connection_tasks.clear()
            self._writers.clear()
            self._conn_to_uuid.clear()

        for task in tasks:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        for writer in writers:
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()

        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

        logger.info("AudioSocket server stopped")

    # ------------------------------------------------------------------
    # Connection handling
    # ------------------------------------------------------------------
    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        conn_id = uuid.uuid4().hex
        peer = writer.get_extra_info("peername")
        logger.info("AudioSocket connection accepted", conn_id=conn_id, peer=peer)

        try:
            sock = writer.get_extra_info("socket")
            if sock:
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except Exception:
            pass

        async with self._lock:
            self._writers[conn_id] = writer
            audiosocket_conn_inc()

        connection_task = asyncio.create_task(self._connection_loop(conn_id, reader, writer))
        async with self._lock:
            self._connection_tasks[conn_id] = connection_task

        try:
            await connection_task
        finally:
            async with self._lock:
                self._connection_tasks.pop(conn_id, None)
                self._writers.pop(conn_id, None)
                self._conn_to_uuid.pop(conn_id, None)
                self._closed_logged.discard(conn_id)
                with contextlib.suppress(ValueError):
                    audiosocket_conn_dec()

    async def _connection_loop(
        self,
        conn_id: str,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        handshake_complete = False
        try:
            while True:
                header = await reader.readexactly(3)
                msg_type = header[0]
                length = int.from_bytes(header[1:], "big")
                payload = b""
                if length:
                    payload = await reader.readexactly(length)

                if not handshake_complete:
                    if msg_type != TYPE_UUID:
                        logger.warning(
                            "Non-UUID frame before handshake",
                            conn_id=conn_id,
                            msg_type=msg_type,
                        )
                        await self._send_error(writer, b"missing-uuid")
                        return

                    uuid_str = self._decode_uuid(payload)
                    if not uuid_str:
                        logger.warning("Invalid UUID payload", conn_id=conn_id)
                        await self._send_error(writer, b"invalid-uuid")
                        return

                    ok = await self._on_uuid(conn_id, uuid_str)
                    if not ok:
                        logger.warning("UUID rejected", conn_id=conn_id, uuid=uuid_str)
                        await self._send_error(writer, b"uuid-rejected")
                        return

                    async with self._lock:
                        self._conn_to_uuid[conn_id] = uuid_str
                    handshake_complete = True
                    logger.info("AudioSocket UUID bound", conn_id=conn_id, uuid=uuid_str)
                    continue

                if msg_type in (TYPE_AUDIO, TYPE_AUDIO_16):
                    if payload:
                        audiosocket_rx_bytes_inc(len(payload))
                        if not self._first_audio_logged.get(conn_id):
                            self._first_audio_logged[conn_id] = True
                            logger.info(
                                "AudioSocket first inbound audio",
                                conn_id=conn_id,
                                bytes=len(payload),
                                msg_type=hex(msg_type),
                            )
                        await self._on_audio(conn_id, payload)
                elif msg_type == TYPE_DTMF:
                    if self._on_dtmf and payload:
                        try:
                            digit = payload.decode("ascii", errors="ignore")
                        except Exception:
                            digit = ""
                        if digit:
                            await self._on_dtmf(conn_id, digit[0])
                elif msg_type in (TYPE_TERMINATE, TYPE_ERROR):
                    logger.info(
                        "AudioSocket terminated by peer",
                        conn_id=conn_id,
                        msg_type=msg_type,
                    )
                    return
                else:
                    logger.debug(
                        "Unknown AudioSocket frame type",
                        conn_id=conn_id,
                        msg_type=msg_type,
                        length=len(payload),
                    )
        except asyncio.IncompleteReadError:
            logger.info("AudioSocket client closed connection", conn_id=conn_id)
        except Exception as exc:
            logger.error("AudioSocket connection error", conn_id=conn_id, error=str(exc), exc_info=True)
        finally:
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()
            if self._on_disconnect:
                await self._on_disconnect(conn_id)

    # ------------------------------------------------------------------
    # Send
    # ------------------------------------------------------------------
    async def send_audio(self, conn_id: str, audio_payload: bytes) -> bool:
        """Send a PCM audio frame to the AudioSocket peer."""
        async with self._lock:
            writer = self._writers.get(conn_id)
        if not writer:
            if conn_id not in self._closed_logged:
                self._closed_logged.add(conn_id)
                logger.debug("Send on closed connection", conn_id=conn_id)
            return False

        frame = bytes([TYPE_AUDIO]) + len(audio_payload).to_bytes(2, "big") + audio_payload
        try:
            writer.write(frame)
            await writer.drain()
            audiosocket_tx_bytes_inc(len(audio_payload))
            return True
        except Exception as exc:
            logger.error("Failed to send audio", conn_id=conn_id, error=str(exc), exc_info=True)
            return False

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    def get_connection_count(self) -> int:
        return len(self._writers)

    def get_uuid_for_conn(self, conn_id: str) -> Optional[str]:
        return self._conn_to_uuid.get(conn_id)

    @property
    def is_bound(self) -> bool:
        return self._server is not None

    async def disconnect(self, conn_id: str) -> None:
        """Proactively close a connection (call cleanup)."""
        async with self._lock:
            writer = self._writers.pop(conn_id, None)
        if not writer:
            return
        writer.close()
        with contextlib.suppress(Exception):
            await writer.wait_closed()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    async def _send_error(self, writer: asyncio.StreamWriter, message: bytes) -> None:
        frame = bytes([TYPE_ERROR]) + len(message).to_bytes(2, "big") + message
        try:
            writer.write(frame)
            await writer.drain()
        except Exception as e:
            logger.debug("Failed to send error frame", error=str(e))

    @staticmethod
    def _decode_uuid(payload: bytes) -> Optional[str]:
        if len(payload) == 16:
            try:
                return str(uuid.UUID(bytes=payload))
            except ValueError:
                return None
        try:
            text = payload.decode("ascii").strip()
            return str(uuid.UUID(text))
        except Exception:
            return None
