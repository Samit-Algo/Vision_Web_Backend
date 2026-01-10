import asyncio
import logging
import os
import subprocess
from dataclasses import dataclass, field
from typing import Dict, Optional, Set

from fastapi import WebSocket

logger = logging.getLogger(__name__)


@dataclass
class _CameraStreamState:
    process: subprocess.Popen
    viewers: Set[WebSocket] = field(default_factory=set)
    task: Optional[asyncio.Task] = None
    last_error: Optional[str] = None
    init_segment: Optional[bytes] = None
    _mp4_parse_buf: bytearray = field(default_factory=bytearray)
    _init_accum: bytearray = field(default_factory=bytearray)
    _init_ready: bool = False


class WsFmp4Service:
    """
    Live streaming service: RTSP -> FFmpeg -> fragmented MP4 (fMP4) -> WebSocket.

    Design goals:
    - 1 FFmpeg process per camera
    - N websocket viewers per camera (broadcast)
    - Prefer passthrough (-c:v copy) for low CPU
    - Drop/disconnect slow clients to avoid memory growth
    """

    def __init__(self) -> None:
        self._streams: Dict[str, _CameraStreamState] = {}
        self._lock = asyncio.Lock()

        # Tunables via env vars (safe defaults)
        self._send_timeout_sec: float = float(os.getenv("WS_STREAM_SEND_TIMEOUT_SEC", "1.0"))
        self._read_chunk_size: int = int(os.getenv("WS_STREAM_READ_CHUNK_SIZE", "4096"))
        self._stop_grace_sec: float = float(os.getenv("WS_STREAM_STOP_GRACE_SEC", "1.0"))

        logger.info("WsFmp4Service initialized")

    def _build_ffmpeg_cmd(self, rtsp_url: str) -> list[str]:
        """
        Build an FFmpeg command that writes fMP4 to stdout.
        Uses stream copy for low CPU. If a camera stream is incompatible with MP4 copy,
        consider adding a transcoding fallback later.
        """
        return [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning",
            # RTSP input stability / latency
            "-rtsp_transport",
            "tcp",
            "-fflags",
            "+discardcorrupt+nobuffer",
            "-flags",
            "low_delay",
            "-use_wallclock_as_timestamps",
            "1",
            # Input
            "-i",
            rtsp_url,
            # Disable audio for simplicity/compatibility (reduce bandwidth too)
            "-an",
            # Low CPU: passthrough video
            "-c:v",
            "copy",
            # Fragmented MP4 suitable for MSE
            "-f",
            "mp4",
            "-movflags",
            "frag_keyframe+empty_moov+default_base_moof",
            # Output to stdout
            "pipe:1",
        ]

    def _start_process(self, camera_id: str, rtsp_url: str) -> subprocess.Popen:
        ffmpeg_cmd = self._build_ffmpeg_cmd(rtsp_url)
        logger.info("Starting WS fMP4 stream for camera %s", camera_id)

        # Note: On Windows, CREATE_NO_WINDOW can interfere with some network setups.
        # Keep default creation flags (same philosophy as your previous HLS service).
        process = subprocess.Popen(
            ffmpeg_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            env=os.environ.copy(),
        )
        return process

    async def add_viewer(self, camera_id: str, websocket: WebSocket, rtsp_url: str) -> None:
        """
        Register a websocket viewer for a camera stream.
        Starts FFmpeg + broadcaster on first viewer.
        """
        init_to_send: Optional[bytes] = None

        async with self._lock:
            state = self._streams.get(camera_id)
            if state is None or state.process.poll() is not None:
                # (Re)start process
                process = self._start_process(camera_id, rtsp_url)
                state = _CameraStreamState(process=process)
                self._streams[camera_id] = state

                # Start background broadcaster task bound to current event loop
                state.task = asyncio.create_task(self._broadcast_loop(camera_id))

            # If we already captured the MP4 init segment (ftyp+moov), send it immediately
            # so new viewers can start decoding even when joining mid-stream.
            if state.init_segment:
                init_to_send = state.init_segment

            state.viewers.add(websocket)

        if init_to_send:
            try:
                await asyncio.wait_for(websocket.send_bytes(init_to_send), timeout=self._send_timeout_sec)
            except Exception:
                # If we can't send init segment, remove viewer to avoid a broken connection lingering
                await self.remove_viewer(camera_id, websocket)
                raise

        logger.info("Viewer added for camera %s. viewers=%d", camera_id, self.get_viewer_count(camera_id))

    async def remove_viewer(self, camera_id: str, websocket: WebSocket) -> None:
        """
        Unregister a websocket viewer.
        Stops FFmpeg when the last viewer disconnects (with a short grace period).
        """
        async with self._lock:
            state = self._streams.get(camera_id)
            if not state:
                return
            state.viewers.discard(websocket)
            remaining = len(state.viewers)

        if remaining == 0:
            # Grace window prevents rapid reconnect flapping from restarting FFmpeg constantly
            await asyncio.sleep(self._stop_grace_sec)
            async with self._lock:
                state = self._streams.get(camera_id)
                if not state or len(state.viewers) > 0:
                    return
            await self.stop_stream(camera_id)

        logger.info("Viewer removed for camera %s. viewers=%d", camera_id, self.get_viewer_count(camera_id))

    async def stop_stream(self, camera_id: str) -> None:
        async with self._lock:
            state = self._streams.get(camera_id)
            if not state:
                return
            process = state.process
            task = state.task

            # Remove from registry early to prevent new viewers binding to the dying process
            del self._streams[camera_id]

        try:
            if task and not task.done():
                task.cancel()
        except Exception:
            pass

        try:
            if process.poll() is None:
                process.terminate()
                await asyncio.to_thread(process.wait, 3)
        except Exception:
            try:
                if process.poll() is None:
                    process.kill()
                    await asyncio.to_thread(process.wait, 2)
            except Exception:
                pass

        logger.info("WS fMP4 stream stopped for camera %s", camera_id)

    async def cleanup_all_streams(self) -> None:
        camera_ids = list(self._streams.keys())
        for camera_id in camera_ids:
            try:
                await self.stop_stream(camera_id)
            except Exception:
                logger.exception("Error stopping WS stream for camera %s", camera_id)

    def is_streaming(self, camera_id: str) -> bool:
        state = self._streams.get(camera_id)
        if not state:
            return False
        return state.process.poll() is None

    def get_viewer_count(self, camera_id: str) -> int:
        state = self._streams.get(camera_id)
        return len(state.viewers) if state else 0

    def get_last_error(self, camera_id: str) -> Optional[str]:
        state = self._streams.get(camera_id)
        return state.last_error if state else None

    async def _broadcast_loop(self, camera_id: str) -> None:
        """
        Reads stdout from FFmpeg and broadcasts bytes to all connected viewers.
        """
        while True:
            async with self._lock:
                state = self._streams.get(camera_id)
                if not state:
                    return
                process = state.process
                viewers = list(state.viewers)

            # If process ended, capture stderr tail and exit
            if process.poll() is not None:
                try:
                    stderr_tail = ""
                    if process.stderr:
                        stderr_tail = (await asyncio.to_thread(process.stderr.read)).decode("utf-8", errors="ignore")[-800:]
                    async with self._lock:
                        state = self._streams.get(camera_id)
                        if state:
                            state.last_error = stderr_tail or f"ffmpeg exited with code {process.returncode}"
                except Exception:
                    pass
                return

            if not process.stdout:
                return

            try:
                chunk = await asyncio.to_thread(process.stdout.read, self._read_chunk_size)
            except asyncio.CancelledError:
                return
            except Exception:
                logger.exception("Error reading ffmpeg stdout for camera %s", camera_id)
                await self.stop_stream(camera_id)
                return

            if not chunk:
                # EOF
                await asyncio.sleep(0.05)
                continue

            # Capture MP4 init segment (ftyp + moov) so viewers joining later can decode.
            # We parse MP4 boxes until we see moov.
            async with self._lock:
                state = self._streams.get(camera_id)
                if state and not state._init_ready:
                    state._mp4_parse_buf.extend(chunk)
                    self._try_extract_init_segment(state)

            if not viewers:
                # Nobody is watching; yield to avoid hot loop until stop_grace triggers.
                await asyncio.sleep(0.05)
                continue

            # Send concurrently with timeout; remove any failing viewers.
            async def _send_one(ws: WebSocket) -> Optional[WebSocket]:
                try:
                    await asyncio.wait_for(ws.send_bytes(chunk), timeout=self._send_timeout_sec)
                    return None
                except Exception:
                    return ws

            results = await asyncio.gather(*(_send_one(ws) for ws in viewers), return_exceptions=False)
            to_remove = [ws for ws in results if ws is not None]

            if to_remove:
                async with self._lock:
                    state = self._streams.get(camera_id)
                    if state:
                        for ws in to_remove:
                            state.viewers.discard(ws)

    def _try_extract_init_segment(self, state: _CameraStreamState) -> None:
        """
        Parse MP4 boxes from the stream until we capture the init segment:
        ftyp + moov (and ignore everything else). Store it in state.init_segment.
        """
        buf = state._mp4_parse_buf

        def read_u32(b: bytes) -> int:
            return int.from_bytes(b, "big", signed=False)

        while True:
            if len(buf) < 8:
                return

            size = read_u32(buf[0:4])
            box_type = bytes(buf[4:8])

            header_len = 8
            if size == 1:
                # 64-bit extended size
                if len(buf) < 16:
                    return
                size = int.from_bytes(buf[8:16], "big", signed=False)
                header_len = 16
            elif size == 0:
                # box extends to end of file/stream (not expected for live)
                return

            if size < header_len or size > 50_000_000:
                # Guard against corruption; stop trying
                return

            if len(buf) < size:
                return

            box = bytes(buf[:size])
            del buf[:size]

            # Keep only ftyp + moov as init segment
            if box_type in (b"ftyp", b"moov"):
                state._init_accum.extend(box)
                if box_type == b"moov":
                    state.init_segment = bytes(state._init_accum)
                    state._init_ready = True
                    # Free memory we no longer need
                    state._mp4_parse_buf.clear()
                    return
