"""
Camera publisher (RTSP → shared memory)
--------------------------------------

This module provides a clean, simple way to read frames from RTSP cameras
and make them available to other processes via shared memory.

HOW IT WORKS:
-------------
1. When a camera is added, CameraPublisher starts automatically
2. It connects to the RTSP stream and decodes frames at the camera's native FPS
3. Each frame is converted to BGR format and stored in shared memory
4. Other processes (agents, live stream) can read the latest frame anytime
5. Only the latest frame is kept (no buffering) for low latency

KEY CONCEPTS:
-------------
- One CameraPublisher per camera (avoids duplicate RTSP connections)
- Runs at camera's native FPS (no artificial limiting)
- Latest-frame-only storage (shared_store[camera_id])
- Independent from agents (agents sample at their own rate)
"""

# -----------------------------------------------------------------------------
# Standard library
# -----------------------------------------------------------------------------
import time
from dataclasses import dataclass
from multiprocessing import Process, Queue
from typing import Any, Dict, Optional, Tuple

# -----------------------------------------------------------------------------
# Third-party
# -----------------------------------------------------------------------------
import numpy as np

try:
    import av  # type: ignore  # PyAV for RTSP
except Exception:  # noqa: BLE001
    av = None  # type: ignore[assignment]

# -----------------------------------------------------------------------------
# Commands and helpers
# -----------------------------------------------------------------------------


@dataclass
class CameraCommand:
    """Command to control the publisher. kind='stop' for graceful shutdown."""
    kind: str  # "stop"


def now_monotonic() -> float:
    """Current monotonic time (for FPS and timing)."""
    return time.monotonic()


# -----------------------------------------------------------------------------
# Camera publisher process
# -----------------------------------------------------------------------------


class CameraPublisher(Process):
    """
    Process that connects to an RTSP camera and publishes the latest frame to shared_store.

    Flow: connect → decode stream → convert each frame to BGR → write to shared_store[camera_id].
    Runs at camera native FPS. Consumers (agents, live stream) read at their own rate.
    """

    def __init__(
        self,
        camera_id: str,
        source_uri: str,
        shared_store: Dict[str, Any],
        command_queue: "Queue",
        reconnect_delay: float = 2.0,
    ) -> None:
        super().__init__(daemon=True)
        self.camera_id = camera_id
        self.source_uri = source_uri
        self.shared_store = shared_store
        self.command_queue: Queue = command_queue
        self.reconnect_delay = reconnect_delay
        self._running = True

    def run(self) -> None:  # type: ignore[override]
        """Main loop: connect to RTSP, decode frames, write latest to shared_store. Reconnects on disconnect."""
        if av is None:
            self.publish_error("PyAV not available; cannot publish RTSP frames")
            return

        while self._running:
            try:
                # 1. Connect to RTSP (TCP for reliability)
                print(f"[publisher {self.camera_id}] 🎥 Connecting to RTSP: {self.source_uri}")
                rtsp_container = av.open(
                    self.source_uri,
                    format="rtsp",
                    options={"rtsp_transport": "tcp", "max_delay": "0"},
                )

                # 2. Get video stream and optional FPS from metadata
                video_stream = rtsp_container.streams.video[0]
                try:
                    camera_fps = float(video_stream.average_rate)
                except (AttributeError, ValueError, TypeError):
                    camera_fps = None
                    print(f"[publisher {self.camera_id}] 📊 Camera FPS: unknown (will measure)")

                # 3. Decode and publish each frame
                frame_index = 0
                stat_last = now_monotonic()
                stat_frames = 0
                last_frame_time = now_monotonic()

                for frame in rtsp_container.decode(video_stream):
                    if not self._running:
                        break
                    self.check_stop_command()

                    # 4. Convert to BGR numpy array
                    frame_bgr = frame.to_ndarray(format="bgr24")
                    height, width = int(frame_bgr.shape[0]), int(frame_bgr.shape[1])
                    frame_index += 1

                    # 5. Measure actual FPS from frame timing
                    now_ts = now_monotonic()
                    if frame_index > 1:
                        frame_delta = now_ts - last_frame_time
                        actual_fps = 1.0 / frame_delta if frame_delta > 0 else 0
                    else:
                        actual_fps = 0
                    last_frame_time = now_ts

                    # 6. Write latest frame to shared memory (overwrites previous)
                    self.shared_store[self.camera_id] = {
                        "shape": (height, width, 3),
                        "dtype": "uint8",
                        "frame_index": frame_index,
                        "ts_monotonic": now_ts,
                        "camera_fps": camera_fps,
                        "actual_fps": actual_fps,
                        "bytes": frame_bgr.tobytes(),
                    }

                    # Log stats once per second
                    stat_frames += 1
                    if (now_ts - stat_last) >= 1.0:
                        print(f"[publisher {self.camera_id}] ⏱️  {stat_frames} frames/s (actual: {actual_fps:.1f} fps)")
                        stat_last = now_ts
                        stat_frames = 0

                print(f"[publisher {self.camera_id}] ⚠️  Connection lost, reconnecting in {self.reconnect_delay}s...")
                time.sleep(self.reconnect_delay)

            except Exception as exc:  # noqa: BLE001
                self.publish_error(f"RTSP error: {exc}")
                print(f"[publisher {self.camera_id}] ⚠️  Error occurred, reconnecting in {self.reconnect_delay}s...")
                time.sleep(self.reconnect_delay)

    def check_stop_command(self) -> None:
        """Non-blocking check for stop command; sets _running = False if received."""
        try:
            while True:
                command: CameraCommand = self.command_queue.get_nowait()
                if command.kind == "stop":
                    print(f"[publisher {self.camera_id}] 🛑 Stop command received")
                    self._running = False
                    break
        except Exception:
            pass

    def publish_error(self, message: str) -> None:
        """Write an error entry to shared_store so consumers know the camera is unavailable."""
        self.shared_store[self.camera_id] = {
            "error": str(message),
            "ts_monotonic": now_monotonic(),
        }
