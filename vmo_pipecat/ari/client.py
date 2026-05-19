"""
ARIClient — single-node Asterisk REST Interface client.

Manages one WebSocket + HTTP session per Asterisk node. Supports supervised
reconnection with exponential back-off (1 s → 60 s).

Used directly for single-node deployments; managed by ARIPool for multi-node.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import glob
import json
import os
import ssl
import time
import uuid
import wave
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import quote

import aiohttp
import websockets
from websockets.asyncio.client import ClientConnection
from websockets.exceptions import ConnectionClosed

from ..observability.log_setup import get_logger

logger = get_logger(__name__)


class ARIClient:
    """A client for interacting with the Asterisk REST Interface (ARI)."""

    def __init__(
        self,
        username: str,
        password: str,
        base_url: str,
        app_name: str,
        ssl_verify: bool = True,
        node_id: str = "default",
    ):
        self.node_id = node_id
        self.username = username
        self.password = password
        self.app_name = app_name
        self.http_url = base_url
        self.ssl_verify = ssl_verify

        if base_url.startswith("https://"):
            ws_scheme = "wss"
            ws_host = base_url.replace("https://", "").split("/")[0]
        else:
            ws_scheme = "ws"
            ws_host = base_url.replace("http://", "").split("/")[0]

        safe_username = quote(username)
        safe_password = quote(password)
        self.ws_url = (
            f"{ws_scheme}://{ws_host}/ari/events"
            f"?api_key={safe_username}:{safe_password}"
            f"&app={app_name}&subscribeAll=true&subscribe=ChannelAudioFrame"
        )

        self.websocket: Optional[ClientConnection] = None
        self.http_session: Optional[aiohttp.ClientSession] = None
        self.running = False
        self._should_reconnect = True
        self._reconnect_attempt = 0
        self._max_reconnect_backoff = 60
        self._connected = False
        self._listener_active = False
        self.event_handlers: Dict[str, List[Callable]] = {}
        self.active_playbacks: Dict[str, str] = {}
        self.audio_frame_handler: Optional[Callable] = None

    # ------------------------------------------------------------------
    # Event registration
    # ------------------------------------------------------------------

    def on_event(self, event_type: str, handler: Callable) -> None:
        """Alias for add_event_handler."""
        self.add_event_handler(event_type, handler)

    def add_event_handler(self, event_type: str, handler: Callable) -> None:
        """Register a handler for a specific ARI event type."""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
        logger.debug(
            "Added event handler",
            node_id=self.node_id,
            event_type=event_type,
            handler=handler.__name__,
        )

    def set_audio_frame_handler(self, handler: Callable) -> None:
        self.audio_frame_handler = handler

    # ------------------------------------------------------------------
    # Connection state
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._connected and self.running and self.websocket is not None

    # ------------------------------------------------------------------
    # Connect / disconnect
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        ws_scheme = "wss" if self.ws_url.startswith("wss://") else "ws"
        http_scheme = "https" if self.http_url.startswith("https://") else "http"
        logger.info(
            "Connecting to ARI...",
            node_id=self.node_id,
            attempt=self._reconnect_attempt + 1,
            http_scheme=http_scheme,
            ws_scheme=ws_scheme,
            http_url=self.http_url,
        )
        self._connected = False
        try:
            ssl_context = None
            if http_scheme == "https":
                ssl_context = ssl.create_default_context()
                if not self.ssl_verify:
                    ssl_context.check_hostname = False
                    ssl_context.verify_mode = ssl.CERT_NONE
                    logger.warning(
                        "SSL certificate verification disabled",
                        node_id=self.node_id,
                    )

            if self.http_session is None or self.http_session.closed:
                connector = aiohttp.TCPConnector(ssl=ssl_context) if ssl_context else None
                self.http_session = aiohttp.ClientSession(
                    auth=aiohttp.BasicAuth(self.username, self.password),
                    connector=connector,
                )

            async with self.http_session.get(
                f"{self.http_url}/asterisk/info"
            ) as response:
                if response.status != 200:
                    raise ConnectionError(
                        f"ARI HTTP probe failed: status {response.status}"
                    )
                logger.info(
                    "ARI HTTP endpoint reachable",
                    node_id=self.node_id,
                    scheme=http_scheme,
                )

            if self.websocket is not None:
                with contextlib.suppress(Exception):
                    await self.websocket.close()
                self.websocket = None

            self.websocket = await websockets.connect(self.ws_url, ssl=ssl_context)
            self.running = True
            self._connected = True
            self._reconnect_attempt = 0
            logger.info(
                "ARI WebSocket connected", node_id=self.node_id, scheme=ws_scheme
            )
        except Exception as exc:
            self._connected = False
            logger.error(
                "Failed to connect to ARI",
                node_id=self.node_id,
                error=str(exc),
                attempt=self._reconnect_attempt + 1,
            )
            if self.http_session and not self.http_session.closed:
                await self.http_session.close()
                self.http_session = None
            raise

    async def disconnect(self) -> None:
        """Disconnect and stop the reconnect supervisor."""
        self._should_reconnect = False
        self._connected = False
        self.running = False
        if self.websocket:
            with contextlib.suppress(Exception):
                await self.websocket.close()
            self.websocket = None
        if self.http_session and not self.http_session.closed:
            await self.http_session.close()
            self.http_session = None
        logger.info("Disconnected from ARI", node_id=self.node_id)

    # ------------------------------------------------------------------
    # Listener loop
    # ------------------------------------------------------------------

    async def start_listening(self) -> None:
        if self._listener_active:
            logger.warning(
                "ARI listener already active; ignoring duplicate start",
                node_id=self.node_id,
            )
            return
        self._listener_active = True
        self._should_reconnect = True
        try:
            await self._listen_with_reconnect()
        finally:
            self._listener_active = False

    async def _mark_disconnected_and_backoff(
        self,
        message: str,
        *,
        level: str = "warning",
        error: Optional[str] = None,
        exc_info: bool = False,
    ) -> bool:
        self._connected = False
        self.running = False
        ws = self.websocket
        self.websocket = None
        if ws is not None:
            with contextlib.suppress(Exception):
                await ws.close()
        if not self._should_reconnect:
            logger.info(f"{message} (shutdown requested).", node_id=self.node_id)
            return False
        self._reconnect_attempt += 1
        backoff = min(2 ** self._reconnect_attempt, self._max_reconnect_backoff)
        log = logger.error if level == "error" else logger.warning
        kwargs: Dict[str, Any] = {
            "node_id": self.node_id,
            "attempt": self._reconnect_attempt,
            "backoff_seconds": backoff,
        }
        if error is not None:
            kwargs["error"] = error
        if exc_info:
            kwargs["exc_info"] = True
        log(message, **kwargs)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + backoff
        while self._should_reconnect:
            remaining = deadline - loop.time()
            if remaining <= 0:
                return True
            await asyncio.sleep(min(remaining, 0.5))
        return False

    async def _listen_with_reconnect(self) -> None:
        while self._should_reconnect:
            if not self.running or not self.websocket:
                try:
                    await self.connect()
                except Exception as exc:
                    should_continue = await self._mark_disconnected_and_backoff(
                        "ARI connection failed, will retry", error=str(exc)
                    )
                    if not should_continue:
                        break
                    continue

            logger.info("Starting ARI event listener", node_id=self.node_id)
            try:
                async for message in self.websocket:
                    try:
                        event_data = json.loads(message)
                        event_type = event_data.get("type")
                        if event_type == "ChannelAudioFrame":
                            channel = event_data.get("channel", {})
                            channel_id = channel.get("id")
                            logger.debug(
                                "ChannelAudioFrame received",
                                node_id=self.node_id,
                                channel_id=channel_id,
                            )
                            asyncio.create_task(
                                self._on_audio_frame(channel, event_data)
                            )
                        if event_type and event_type in self.event_handlers:
                            for handler in self.event_handlers[event_type]:
                                asyncio.create_task(handler(event_data))
                    except json.JSONDecodeError:
                        logger.warning(
                            "Failed to decode ARI event JSON",
                            node_id=self.node_id,
                            message=message,
                        )

                should_continue = await self._mark_disconnected_and_backoff(
                    "ARI WebSocket listener ended, will reconnect"
                )
                if not should_continue:
                    break

            except ConnectionClosed:
                should_continue = await self._mark_disconnected_and_backoff(
                    "ARI WebSocket connection closed, will reconnect"
                )
                if not should_continue:
                    break

            except Exception as exc:
                should_continue = await self._mark_disconnected_and_backoff(
                    "ARI listener error, will reconnect",
                    level="error",
                    error=str(exc),
                    exc_info=True,
                )
                if not should_continue:
                    break

        logger.info("ARI reconnect supervisor stopped", node_id=self.node_id)

    # ------------------------------------------------------------------
    # HTTP commands
    # ------------------------------------------------------------------

    async def send_command(
        self,
        method: str,
        resource: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        tolerate_statuses: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        url = f"{self.http_url}/{resource}"
        if params and "channelVars" in params:
            channel_vars = params.pop("channelVars")
            if data is None:
                data = {}
            data["channelVars"] = channel_vars
        try:
            async with self.http_session.request(
                method, url, json=data, params=params
            ) as response:
                if response.status >= 400:
                    reason = await response.text()
                    if (
                        int(response.status) == 404
                        and str(method).upper() == "GET"
                        and "/channels/" in f"/{resource}"
                        and str(resource).endswith("/variable")
                        and "Provided variable was not found" in reason
                    ):
                        logger.debug(
                            "ARI channel variable not found (benign)",
                            node_id=self.node_id,
                            method=method,
                            url=url,
                            status=response.status,
                        )
                        return {"status": response.status, "reason": reason}
                    if tolerate_statuses and response.status in tolerate_statuses:
                        logger.debug(
                            "ARI command tolerated non-2xx",
                            node_id=self.node_id,
                            method=method,
                            url=url,
                            status=response.status,
                        )
                    else:
                        logger.error(
                            "ARI command failed",
                            node_id=self.node_id,
                            method=method,
                            url=url,
                            status=response.status,
                            reason=reason,
                        )
                    return {"status": response.status, "reason": reason}
                if response.status == 204:
                    return {"status": response.status}
                return await response.json()
        except aiohttp.ClientError as exc:
            logger.error("ARI HTTP request failed", node_id=self.node_id, exc_info=True)
            return {"status": 500, "reason": str(exc)}

    async def originate_channel(
        self,
        *,
        endpoint: str,
        app: str,
        app_args: str = "",
        timeout: int = 60,
        caller_id: str = "",
        channel_vars: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "endpoint": str(endpoint),
            "app": str(app),
            "timeout": str(int(timeout)),
        }
        if app_args:
            params["appArgs"] = str(app_args)
        if caller_id:
            params["callerId"] = str(caller_id)
        if channel_vars:
            params["channelVars"] = channel_vars
        return await self.send_command("POST", "channels", params=params)

    async def continue_in_dialplan(
        self,
        channel_id: str,
        *,
        context: str,
        extension: str = "s",
        priority: int = 1,
        label: Optional[str] = None,
    ) -> bool:
        params: Dict[str, Any] = {
            "context": str(context),
            "extension": str(extension),
            "priority": str(int(priority)),
        }
        if label:
            params["label"] = str(label)
        resp = await self.send_command(
            "POST", f"channels/{channel_id}/continue", params=params
        )
        status = resp.get("status") if isinstance(resp, dict) else None
        if status is not None and int(status) >= 400:
            return False
        return True

    async def answer_channel(self, channel_id: str) -> None:
        logger.info("Answering channel", node_id=self.node_id, channel_id=channel_id)
        await self.send_command("POST", f"channels/{channel_id}/answer")

    async def hangup_channel(self, channel_id: str) -> None:
        logger.info("Hanging up channel", node_id=self.node_id, channel_id=channel_id)
        response = await self.send_command(
            "DELETE", f"channels/{channel_id}", tolerate_statuses=[404]
        )
        if response and response.get("status") == 404:
            logger.debug(
                "Channel hangup 404 — already gone",
                node_id=self.node_id,
                channel_id=channel_id,
            )

    async def execute_application(
        self, channel_id: str, app_name: str, app_data: str
    ) -> bool:
        try:
            response = await self.send_command(
                "POST",
                f"channels/{channel_id}/applications/{app_name}",
                data={"app": app_name, "appArgs": app_data},
            )
            return bool(response)
        except Exception:
            logger.error(
                "Error executing application",
                node_id=self.node_id,
                channel_id=channel_id,
                app_name=app_name,
                exc_info=True,
            )
            return False

    async def play_media(
        self, channel_id: str, media_uri: str
    ) -> Optional[Dict[str, Any]]:
        return await self.send_command(
            "POST", f"channels/{channel_id}/play", data={"media": media_uri}
        )

    async def play_sound(
        self, channel_id: str, sound_file: str
    ) -> Optional[Dict[str, Any]]:
        media_uri = (sound_file or "").strip()
        if not media_uri:
            return None
        if not any(
            media_uri.startswith(p) for p in ("sound:", "file:", "recording:")
        ):
            media_uri = f"sound:{media_uri}"
        return await self.play_media(channel_id, media_uri)

    async def play_media_on_channel_with_id(
        self, channel_id: str, media_uri: str, playback_id: str
    ) -> bool:
        try:
            response = await self.send_command(
                "POST",
                f"channels/{channel_id}/play",
                data={"media": media_uri, "playbackId": playback_id},
            )
            return bool(response and response.get("id") == playback_id)
        except Exception:
            logger.error(
                "Error starting channel playback with deterministic ID",
                node_id=self.node_id,
                channel_id=channel_id,
                exc_info=True,
            )
            return False

    async def set_channel_var(
        self, channel_id: str, variable: str, value: str = ""
    ) -> bool:
        try:
            resp = await self.send_command(
                "POST",
                f"channels/{channel_id}/variable",
                data={"variable": variable, "value": value},
            )
            return resp is not None
        except Exception:
            logger.error(
                "Failed to set channel variable",
                node_id=self.node_id,
                channel_id=channel_id,
                variable=variable,
                exc_info=True,
            )
            return False

    async def create_bridge(
        self, bridge_type: str = "mixing"
    ) -> Optional[str]:
        try:
            response = await self.send_command(
                "POST",
                "bridges",
                data={"type": bridge_type, "name": f"bridge_{uuid.uuid4().hex[:8]}"},
            )
            if response.get("id"):
                logger.info(
                    "Bridge created",
                    node_id=self.node_id,
                    bridge_id=response["id"],
                    bridge_type=bridge_type,
                )
                return response["id"]
            logger.error(
                "Failed to create bridge", node_id=self.node_id, response=response
            )
            return None
        except Exception as exc:
            logger.error("Error creating bridge", node_id=self.node_id, error=str(exc))
            return None

    async def stop_playback(self, playback_id: str) -> bool:
        try:
            response = await self.send_command("DELETE", f"playbacks/{playback_id}")
            status = response.get("status") if isinstance(response, dict) else None
            if status is not None:
                if 200 <= int(status) < 300:
                    return True
                logger.debug(
                    "Failed to stop playback (may already be finished)",
                    node_id=self.node_id,
                    playback_id=playback_id,
                )
                return False
            return True
        except Exception:
            logger.error(
                "Error stopping playback",
                node_id=self.node_id,
                playback_id=playback_id,
                exc_info=True,
            )
            return False

    async def record_channel(
        self,
        channel_id: str,
        name: str,
        format: str = "wav",
        if_exists: str = "overwrite",
        max_duration_seconds: int = 180,
        max_silence_seconds: int = 0,
        beep: bool = False,
        terminate_on: str = "none",
    ) -> bool:
        try:
            payload = {
                "name": str(name),
                "format": str(format),
                "ifExists": str(if_exists),
                "maxDurationSeconds": str(int(max_duration_seconds)),
                "maxSilenceSeconds": str(int(max_silence_seconds)),
                "beep": "true" if bool(beep) else "false",
                "terminateOn": str(terminate_on),
            }
            response = await self.send_command(
                "POST", f"channels/{channel_id}/record", params=payload
            )
            status = response.get("status") if isinstance(response, dict) else None
            if status is not None and not (200 <= int(status) < 300):
                logger.error(
                    "Failed to start ARI channel recording",
                    node_id=self.node_id,
                    channel_id=channel_id,
                )
                return False
            return True
        except Exception:
            logger.error(
                "Error starting ARI channel recording",
                node_id=self.node_id,
                channel_id=channel_id,
                exc_info=True,
            )
            return False

    async def add_channel_to_bridge(self, bridge_id: str, channel_id: str) -> bool:
        try:
            response = await self.send_command(
                "POST",
                f"bridges/{bridge_id}/addChannel",
                data={"channel": channel_id},
            )
            status = response.get("status") if isinstance(response, dict) else None
            if status is not None:
                if 200 <= int(status) < 300:
                    return True
                reason = str(response.get("reason", "") or "")
                if int(status) in (409, 422) and "already" in reason.lower() and "bridge" in reason.lower():
                    return True
                logger.error(
                    "Failed to add channel to bridge",
                    node_id=self.node_id,
                    bridge_id=bridge_id,
                    channel_id=channel_id,
                    status=status,
                )
                return False
            return True
        except Exception as exc:
            logger.error(
                "Error adding channel to bridge",
                node_id=self.node_id,
                bridge_id=bridge_id,
                channel_id=channel_id,
                error=str(exc),
            )
            return False

    async def remove_channel_from_bridge(
        self, bridge_id: str, channel_id: str
    ) -> bool:
        try:
            response = await self.send_command(
                "POST",
                f"bridges/{bridge_id}/removeChannel",
                data={"channel": channel_id},
            )
            status = response.get("status") if isinstance(response, dict) else None
            if status is not None:
                if 200 <= int(status) < 300:
                    return True
                logger.error(
                    "Failed to remove channel from bridge",
                    node_id=self.node_id,
                    bridge_id=bridge_id,
                    channel_id=channel_id,
                    status=status,
                )
                return False
            return True
        except Exception as exc:
            logger.error(
                "Error removing channel from bridge",
                node_id=self.node_id,
                bridge_id=bridge_id,
                channel_id=channel_id,
                error=str(exc),
            )
            return False

    async def destroy_bridge(self, bridge_id: str) -> bool:
        try:
            response = await self.send_command(
                "DELETE", f"bridges/{bridge_id}", tolerate_statuses=[404]
            )
            status = response.get("status") if isinstance(response, dict) else None
            if status is not None:
                if 200 <= int(status) < 300:
                    return True
                if int(status) == 404:
                    return True
                logger.error(
                    "Failed to destroy bridge",
                    node_id=self.node_id,
                    bridge_id=bridge_id,
                    status=status,
                )
                return False
            return True
        except Exception as exc:
            logger.error(
                "Error destroying bridge",
                node_id=self.node_id,
                bridge_id=bridge_id,
                error=str(exc),
            )
            return False

    async def is_channel_active(self, channel_id: str) -> bool:
        try:
            result = await self.send_command("GET", f"channels/{channel_id}")
            if result and result.get("id") == channel_id:
                state = result.get("state", "")
                return state in ["Up", "Ring", "Ringing", "Dialing"]
            return False
        except Exception:
            return False

    async def validate_channel_for_playback(self, channel_id: str) -> bool:
        try:
            if not await self.is_channel_active(channel_id):
                return False
            result = await self.send_command("GET", f"channels/{channel_id}")
            if not result:
                return False
            return result.get("state", "") == "Up"
        except Exception:
            return False

    async def play_audio_response(self, channel_id: str, audio_data: bytes) -> None:
        if not await self.validate_channel_for_playback(channel_id):
            logger.warning(
                "Channel validation failed — skipping audio playback",
                node_id=self.node_id,
                channel_id=channel_id,
            )
            return
        unique_filename = f"response-{uuid.uuid4()}.ulaw"
        container_path = f"/mnt/asterisk_media/ai-generated/{unique_filename}"
        asterisk_media_uri = f"sound:ai-generated/{unique_filename[:-5]}"
        try:
            with open(container_path, "wb") as f:
                f.write(audio_data)
            playback = await self.play_media(channel_id, asterisk_media_uri)
            if playback and "id" in playback:
                self.active_playbacks[playback["id"]] = container_path
        except Exception as exc:
            logger.error(
                "Failed to play audio file",
                node_id=self.node_id,
                channel_id=channel_id,
                error=str(exc),
                exc_info=True,
            )

    async def _on_playback_finished(self, event: dict) -> None:
        playback_id = event.get("playback", {}).get("id")
        file_path = self.active_playbacks.pop(playback_id, None)
        if file_path:
            await asyncio.sleep(2.0)
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except OSError:
                    logger.error(
                        "Error deleting audio file",
                        node_id=self.node_id,
                        file_path=file_path,
                        exc_info=True,
                    )
        if hasattr(self, "engine") and hasattr(self.engine, "_on_playback_finished"):
            await self.engine._on_playback_finished(event)

    async def cleanup_call_files(self, channel_id: str) -> None:
        ai_generated_dir = "/mnt/asterisk_media/ai-generated"
        if os.path.exists(ai_generated_dir):
            pattern = os.path.join(ai_generated_dir, "response-*.ulaw")
            for file_path in glob.glob(pattern):
                try:
                    if os.path.getmtime(file_path) < (time.time() - 30):
                        os.remove(file_path)
                except OSError:
                    pass

    async def _on_audio_frame(self, channel: dict, event: dict) -> None:
        try:
            frame_data = event.get("frame", {})
            audio_payload = frame_data.get("data", "")
            if audio_payload:
                audio_data = base64.b64decode(audio_payload)
                if self.audio_frame_handler:
                    await self.audio_frame_handler(audio_data)
        except Exception as exc:
            logger.error(
                "Error processing audio frame",
                node_id=self.node_id,
                error=str(exc),
                exc_info=True,
            )

    async def handle_audio_frame(
        self, event_data: dict, audio_handler: Callable
    ) -> None:
        channel = event_data.get("channel", {})
        channel_id = channel.get("id")
        audio_data = event_data.get("audio", {})
        if channel_id and audio_data:
            audio_payload = audio_data.get("data")
            if audio_payload:
                try:
                    raw_audio = base64.b64decode(audio_payload)
                    await audio_handler(channel_id, raw_audio)
                except Exception as exc:
                    logger.error(
                        "Error processing audio frame",
                        node_id=self.node_id,
                        error=str(exc),
                    )

    async def handle_dtmf_received(
        self, event_data: dict, dtmf_handler: Callable
    ) -> None:
        channel = event_data.get("channel", {})
        channel_id = channel.get("id")
        digit = event_data.get("digit")
        if channel_id and digit:
            await dtmf_handler(channel_id, digit)

    async def create_external_media_channel(
        self,
        app: str,
        external_host: str,
        format: str = "ulaw",
        direction: str = "both",
        encapsulation: str = "rtp",
    ) -> Optional[Dict[str, Any]]:
        try:
            response = await self.send_command(
                "POST",
                "channels/externalMedia",
                data={
                    "app": app,
                    "external_host": external_host,
                    "format": format,
                    "direction": direction,
                    "encapsulation": encapsulation,
                },
            )
            if response and response.get("id"):
                return response
            logger.error(
                "Failed to create External Media channel",
                node_id=self.node_id,
                response=response,
            )
            return None
        except Exception as exc:
            logger.error(
                "Error creating External Media channel",
                node_id=self.node_id,
                external_host=external_host,
                error=str(exc),
            )
            return None

    async def create_external_media(
        self,
        external_host: str,
        external_port: int,
        fmt: str = "ulaw",
        direction: str = "both",
    ) -> Optional[str]:
        response = await self.create_external_media_channel(
            app=self.app_name,
            external_host=f"{external_host}:{external_port}",
            format=fmt,
            direction=direction,
        )
        if response and response.get("id"):
            return response["id"]
        return None

    async def play_audio_via_bridge(
        self, bridge_id: str, media_uri: str
    ) -> Optional[str]:
        try:
            response = await self.send_command(
                "POST", f"bridges/{bridge_id}/play", data={"media": media_uri}
            )
            if response and response.get("id"):
                return response["id"]
            return None
        except Exception:
            logger.error(
                "Error starting bridge playback",
                node_id=self.node_id,
                bridge_id=bridge_id,
                exc_info=True,
            )
            return None

    async def play_media_on_bridge_with_id(
        self, bridge_id: str, media_uri: str, playback_id: str
    ) -> bool:
        try:
            response = await self.send_command(
                "POST",
                f"bridges/{bridge_id}/play",
                data={"media": media_uri, "playbackId": playback_id},
            )
            return bool(response and response.get("id") == playback_id)
        except Exception:
            logger.error(
                "Error starting bridge playback with deterministic ID",
                node_id=self.node_id,
                bridge_id=bridge_id,
                exc_info=True,
            )
            return False

    async def play_audio_file(self, channel_id: str, file_path: str) -> bool:
        try:
            for _ in range(15):
                if (
                    os.path.exists(file_path)
                    and os.access(file_path, os.R_OK)
                    and os.path.getsize(file_path) > 0
                ):
                    break
                await asyncio.sleep(0.1)
            if not os.path.exists(file_path) or not os.access(file_path, os.R_OK):
                logger.error(
                    "Audio file not accessible",
                    node_id=self.node_id,
                    file_path=file_path,
                )
                return False
            result = await self.send_command(
                "POST",
                f"channels/{channel_id}/play",
                data={"media": f"sound:{file_path}"},
            )
            return bool(result)
        except Exception as exc:
            logger.error(
                "Error playing audio file",
                node_id=self.node_id,
                file_path=file_path,
                error=str(exc),
            )
            return False

    async def create_audio_file_from_ulaw(
        self, ulaw_data: bytes, sample_rate: int = 8000
    ) -> str:
        try:
            import audioop

            pcm_data = audioop.ulaw2lin(ulaw_data, 2)
            timestamp = int(time.time() * 1000)
            filename = f"audio_{timestamp}_{len(pcm_data)}.wav"
            temp_file_path = f"/tmp/asterisk-audio/{filename}"
            try:
                os.makedirs("/tmp/asterisk-audio", mode=0o700, exist_ok=True)
            except Exception:
                pass
            with wave.open(temp_file_path, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(pcm_data)
            os.chmod(temp_file_path, 0o600)
            await asyncio.to_thread(os.sync)
            for _ in range(10):
                if (
                    os.path.exists(temp_file_path)
                    and os.access(temp_file_path, os.R_OK)
                    and os.path.getsize(temp_file_path) > 0
                ):
                    return temp_file_path
                await asyncio.sleep(0.1)
            return ""
        except Exception as exc:
            logger.error(
                "Error creating audio file from ulaw",
                node_id=self.node_id,
                error=str(exc),
            )
            return ""

    async def cleanup_audio_file(self, file_path: str, delay: float = 5.0) -> None:
        try:
            await asyncio.sleep(delay)
            if os.path.exists(file_path):
                os.unlink(file_path)
        except Exception as exc:
            logger.error(
                "Error cleaning up audio file",
                node_id=self.node_id,
                file_path=file_path,
                error=str(exc),
            )

    async def stop_audio_streaming(self, channel_id: str) -> bool:
        media_info = getattr(self, "active_media_channels", {}).pop(channel_id, None)
        if not media_info:
            return True
        try:
            await self.destroy_bridge(media_info["bridge_id"])
            await self.hangup_channel(media_info["media_channel_id"])
            return True
        except Exception:
            logger.error(
                "Error during audio streaming cleanup",
                node_id=self.node_id,
                exc_info=True,
            )
            return False
