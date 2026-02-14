"""
Agent processed-frame streaming: shared_store -> FFmpeg (rawvideo) -> fMP4 -> WebSocket.

Reads processed frames (with detections already drawn) from shared_store[agent_id],
feeds them to FFmpeg as rawvideo, and broadcasts fMP4 bytes to viewers.
Same format as raw live stream (fMP4) so the client can use one code path.
"""

import asyncio
import logging
import os
import subprocess
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set

from fastapi import WebSocket

from .fmp4_common import try_extract_init_segment

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Tunables (env)
# -----------------------------------------------------------------------------
def _send_timeout_sec() -> float:
    return float(os.getenv("WS_STREAM_SEND_TIMEOUT_SEC", "1.0"))


def _read_chunk_size() -> int:
    return int(os.getenv("WS_STREAM_READ_CHUNK_SIZE", "4096"))


def _stop_grace_sec() -> float:
    return float(os.getenv("WS_STREAM_STOP_GRACE_SEC", "1.0"))


# -----------------------------------------------------------------------------
# State per agent stream
# -----------------------------------------------------------------------------
@dataclass
class _AgentStreamState:
    process: Optional[subprocess.Popen] = None
    viewers: Set[WebSocket] = field(default_factory=set)
    feeder_task: Optional[asyncio.Task] = None
    broadcast_task: Optional[asyncio.Task] = None
    last_error: Optional[str] = None
    init_segment: Optional[bytes] = None
    _mp4_parse_buf: bytearray = field(default_factory=bytearray)
    _init_accum: bytearray = field(default_factory=bytearray)
    _init_ready: bool = False


# -----------------------------------------------------------------------------
# Service
# -----------------------------------------------------------------------------
class ProcessedFrameStreamService:
    """
    Streams processed video frames (with detections drawn) as fMP4 over WebSocket.

    - One FFmpeg process per agent (when there is at least one viewer).
    - Frames are read from shared_store[agent_id] and fed to FFmpeg via stdin.
    - Output is fMP4, same as raw live stream, for consistent client handling.
    """

    def __init__(self) -> None:
        self._streams: Dict[str, _AgentStreamState] = {}
        self._lock = asyncio.Lock()
        logger.info("ProcessedFrameStreamService initialized")

    def _build_ffmpeg_cmd(self, width: int, height: int, fps: float) -> list:
        """Build FFmpeg command: rawvideo stdin -> fMP4 stdout."""
        fps_int = max(1, min(30, int(round(fps))))
        return [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-f", "rawvideo",
            "-pixel_format", "bgr24",
            "-s", f"{width}x{height}",
            "-r", str(fps_int),
            "-i", "pipe:0",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-tune", "zerolatency",
            "-pix_fmt", "yuv420p",
            "-f", "mp4",
            "-movflags", "frag_keyframe+empty_moov+default_base_moof",
            "pipe:1",
        ]

    def _start_process(self, agent_id: str, width: int, height: int, fps: float) -> subprocess.Popen:
        cmd = self._build_ffmpeg_cmd(width, height, fps)
        logger.info("Starting processed fMP4 stream for agent %s (%dx%d @ %s fps)", agent_id, width, height, fps)
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            env=os.environ.copy(),
        )
        return process

    async def add_viewer(
        self,
        agent_id: str,
        websocket: WebSocket,
        shared_store: Any,
    ) -> None:
        """
        Register a viewer for an agent's processed-frame stream.
        Starts feeder (and FFmpeg when first frame is available) on first viewer.
        """
        init_to_send: Optional[bytes] = None

        async with self._lock:
            state = self._streams.get(agent_id)
            if state is None:
                state = _AgentStreamState()
                self._streams[agent_id] = state
                state.feeder_task = asyncio.create_task(
                    self._feeder_loop(agent_id, shared_store)
                )

            if state.init_segment:
                init_to_send = state.init_segment

            state.viewers.add(websocket)

        if init_to_send:
            try:
                await asyncio.wait_for(
                    websocket.send_bytes(init_to_send),
                    timeout=_send_timeout_sec(),
                )
            except Exception:
                await self.remove_viewer(agent_id, websocket)
                raise

        logger.info(
            "Viewer added for agent %s. viewers=%d",
            agent_id,
            self.get_viewer_count(agent_id),
        )

    async def remove_viewer(self, agent_id: str, websocket: WebSocket) -> None:
        """Unregister a viewer. Stops FFmpeg when the last viewer disconnects."""
        async with self._lock:
            state = self._streams.get(agent_id)
            if not state:
                return
            state.viewers.discard(websocket)
            remaining = len(state.viewers)

        if remaining == 0:
            await asyncio.sleep(_stop_grace_sec())
            async with self._lock:
                state = self._streams.get(agent_id)
                if not state or len(state.viewers) > 0:
                    return
            await self.stop_stream(agent_id)

        logger.info(
            "Viewer removed for agent %s. viewers=%d",
            agent_id,
            self.get_viewer_count(agent_id),
        )

    async def stop_stream(self, agent_id: str) -> None:
        """Stop the stream for an agent and clean up FFmpeg and tasks."""
        async with self._lock:
            state = self._streams.get(agent_id)
            if not state:
                return
            process = state.process
            feeder_task = state.feeder_task
            broadcast_task = state.broadcast_task
            del self._streams[agent_id]

        for task in (feeder_task, broadcast_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        if process and process.poll() is None:
            try:
                process.terminate()
                await asyncio.to_thread(process.wait, 3)
            except Exception:
                try:
                    if process.poll() is None:
                        process.kill()
                        await asyncio.to_thread(process.wait, 2)
                except Exception:
                    pass

        logger.info("Processed fMP4 stream stopped for agent %s", agent_id)

    async def cleanup_all_streams(self) -> None:
        """Stop all agent streams (e.g. on app shutdown)."""
        agent_ids = list(self._streams.keys())
        for agent_id in agent_ids:
            try:
                await self.stop_stream(agent_id)
            except Exception:
                logger.exception("Error stopping processed stream for agent %s", agent_id)

    def is_streaming(self, agent_id: str) -> bool:
        state = self._streams.get(agent_id)
        if not state or not state.process:
            return False
        return state.process.poll() is None

    def get_viewer_count(self, agent_id: str) -> int:
        state = self._streams.get(agent_id)
        return len(state.viewers) if state else 0

    def get_last_error(self, agent_id: str) -> Optional[str]:
        state = self._streams.get(agent_id)
        return state.last_error if state else None

    async def _feeder_loop(self, agent_id: str, shared_store: Any) -> None:
        """
        Wait for first frame from shared_store[agent_id], start FFmpeg and broadcast task,
        then keep writing frame bytes to FFmpeg stdin.
        When no new frame is available (e.g. RTSP reconnecting), repeat last frame so
        the stream does not stall and the client does not show endless loading.
        """
        width, height, fps = 0, 0, 5.0
        process: Optional[subprocess.Popen] = None
        last_frame_bytes: Optional[bytes] = None
        frame_interval = 1.0 / 5.0  # will be updated from first frame

        while True:
            async with self._lock:
                state = self._streams.get(agent_id)
                if not state:
                    return
                viewers = list(state.viewers)

            if not viewers:
                await asyncio.sleep(0.1)
                continue

            entry = None
            try:
                entry = shared_store.get(agent_id)
            except Exception:
                pass

            frame_bytes: Optional[bytes] = None
            shape: Optional[list] = None

            if isinstance(entry, dict):
                frame_bytes = entry.get("bytes")
                shape = entry.get("shape")

            if frame_bytes and shape and len(shape) >= 2:
                try:
                    h, w = int(shape[0]), int(shape[1])
                except (TypeError, ValueError):
                    h, w = 0, 0

                if process is None and w > 0 and h > 0:
                    fps_val = entry.get("actual_fps") or entry.get("camera_fps") or 5
                    try:
                        fps_val = float(fps_val)
                    except (TypeError, ValueError):
                        fps_val = 5.0
                    fps = max(1, min(30, fps_val))
                    frame_interval = 1.0 / max(1, fps)

                    process = self._start_process(agent_id, w, h, fps)

                    async with self._lock:
                        state = self._streams.get(agent_id)
                        if not state:
                            if process.poll() is None:
                                process.terminate()
                            return
                        state.process = process
                        state.broadcast_task = asyncio.create_task(
                            self._broadcast_loop(agent_id)
                        )

                if process and process.poll() is None:
                    last_frame_bytes = frame_bytes
                    try:
                        process.stdin.write(frame_bytes)
                        process.stdin.flush()
                    except Exception as e:
                        logger.warning("Feeder write error for agent %s: %s", agent_id, e)
                        async with self._lock:
                            state = self._streams.get(agent_id)
                            if state:
                                state.last_error = str(e)
                        return
                    await asyncio.sleep(frame_interval)
                    continue

            if process is not None and process.poll() is None and last_frame_bytes:
                # No new frame (e.g. RTSP reconnecting). Repeat last frame so stream does not stall.
                try:
                    process.stdin.write(last_frame_bytes)
                    process.stdin.flush()
                except Exception as e:
                    logger.warning("Feeder write error for agent %s: %s", agent_id, e)
                    async with self._lock:
                        state = self._streams.get(agent_id)
                        if state:
                            state.last_error = str(e)
                    return
                await asyncio.sleep(frame_interval)
                continue

            await asyncio.sleep(0.25)

    async def _broadcast_loop(self, agent_id: str) -> None:
        """Read fMP4 from FFmpeg stdout and send to all viewers. Reuses shared init-segment parsing."""
        while True:
            async with self._lock:
                state = self._streams.get(agent_id)
                if not state:
                    return
                process = state.process
                viewers = list(state.viewers)

            if not process or process.poll() is not None:
                if process and process.stderr:
                    try:
                        stderr_tail = (
                            await asyncio.to_thread(process.stderr.read)
                        ).decode("utf-8", errors="ignore")[-800:]
                        async with self._lock:
                            s = self._streams.get(agent_id)
                            if s:
                                s.last_error = stderr_tail or f"ffmpeg exited {process.returncode}"
                    except Exception:
                        pass
                return

            if not process.stdout:
                return

            try:
                chunk = await asyncio.to_thread(
                    process.stdout.read, _read_chunk_size()
                )
            except asyncio.CancelledError:
                return
            except Exception:
                logger.exception("Error reading ffmpeg stdout for agent %s", agent_id)
                await self.stop_stream(agent_id)
                return

            if not chunk:
                await asyncio.sleep(0.05)
                continue

            async with self._lock:
                state = self._streams.get(agent_id)
                if state and not state._init_ready:
                    state._mp4_parse_buf.extend(chunk)
                    ready, init_segment = try_extract_init_segment(
                        state._mp4_parse_buf, state._init_accum
                    )
                    if ready:
                        state.init_segment = init_segment
                        state._init_ready = True
                        state._mp4_parse_buf.clear()

            if not viewers:
                await asyncio.sleep(0.05)
                continue

            async def _send_one(ws: WebSocket):
                try:
                    await asyncio.wait_for(
                        ws.send_bytes(chunk), timeout=_send_timeout_sec()
                    )
                    return None
                except Exception:
                    return ws

            results = await asyncio.gather(
                *(_send_one(ws) for ws in viewers),
                return_exceptions=False,
            )
            to_remove = [ws for ws in results if ws is not None]
            if to_remove:
                async with self._lock:
                    state = self._streams.get(agent_id)
                    if state:
                        for ws in to_remove:
                            state.viewers.discard(ws)
