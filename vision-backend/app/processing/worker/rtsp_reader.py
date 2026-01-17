"""
Threaded RTSP reader using PyAV for stable H264/H265 over RTSP.
"""
import threading
import time
from typing import Optional

import av  # type: ignore
import numpy as np


class RTSPReader:
    """
    Background reader that continuously decodes frames from an RTSP source
    using PyAV/FFmpeg and exposes the latest BGR frame via get_frame().
    """

    def __init__(self, url: str, reconnect_delay: float = 2.0) -> None:
        self.url = url
        self.reconnect_delay = reconnect_delay
        self._frame: Optional[np.ndarray] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join()

    def _reader_loop(self) -> None:
        while self._running:
            try:
                # Low-latency TCP transport; disable max_delay jitter buffer
                container = av.open(
                    self.url,
                    format="rtsp",
                    options={
                        "rtsp_transport": "tcp",
                        "max_delay": "0",
                    },
                )

                stream = container.streams.video[0]
                # Skip non-key frames if decoder struggles; stabilizes on some cameras
                stream.codec_context.skip_frame = "NONKEY"  # type: ignore[attr-defined]

                for frame in container.decode(stream):
                    if not self._running:
                        break
                    # Convert to OpenCV-compatible BGR
                    img = frame.to_ndarray(format="bgr24")
                    self._frame = img
            except Exception as exc:  # noqa: BLE001
                print(f"⚠️ RTSP reconnecting due to: {exc}")
                time.sleep(self.reconnect_delay)

    def get_frame(self) -> Optional[np.ndarray]:
        return self._frame


