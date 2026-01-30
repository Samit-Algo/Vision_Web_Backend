"""
Camera Frame Publisher
======================

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
import time
from dataclasses import dataclass
from multiprocessing import Process, Queue
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
	# PyAV for robust RTSP decoding
	import av  # type: ignore
except Exception:  # noqa: BLE001
	av = None  # type: ignore[assignment]


@dataclass
class CameraCommand:
	"""
	Simple command to control CameraPublisher.
	
	Currently only supports "stop" command to gracefully shutdown.
	"""
	kind: str  # "stop"
	# No value field needed - we removed FPS control


def _now_monotonic() -> float:
	"""Get current monotonic time (for accurate timing measurements)"""
	return time.monotonic()


class CameraPublisher(Process):
	"""
	Camera Frame Publisher - Reads RTSP stream and publishes frames to shared memory.
	
	FLOW:
	-----
	1. Connect to RTSP camera stream
	2. Decode frames as fast as camera provides them (native FPS)
	3. Convert each frame to BGR numpy array
	4. Store latest frame in shared_store[camera_id]
	5. Repeat until stopped
	
	IMPORTANT:
	----------
	- Runs at camera's native FPS (7fps, 10fps, 30fps - whatever camera provides)
	- No FPS limiting or artificial delays
	- Only stores latest frame (overwrites previous)
	- Agents and live stream read independently at their own rates
	"""

	def __init__(
		self,
		camera_id: str,
		source_uri: str,
		shared_store: "Dict[str, Any]",  # proxy from multiprocessing.Manager
		command_queue: "Queue",
		reconnect_delay: float = 2.0,
	) -> None:
		"""
		Initialize CameraPublisher.
		
		Args:
			camera_id: Unique identifier for this camera
			source_uri: RTSP URL (e.g., "rtsp://192.168.1.100:554/stream")
			shared_store: Shared memory dict where frames will be stored
			command_queue: Queue for receiving stop commands
			reconnect_delay: Seconds to wait before reconnecting on error
		"""
		super().__init__(daemon=True)
		self.camera_id = camera_id
		self.source_uri = source_uri
		self.shared_store = shared_store
		self.command_queue: "Queue[CameraCommand]" = command_queue
		self.reconnect_delay = reconnect_delay
		self._running = True

	def run(self) -> None:  # type: ignore[override]
		"""
		Main loop: Connect to camera and continuously publish frames.
		
		This runs in a separate process, so it doesn't block other operations.
		"""
		if av is None:
			self._publish_error("PyAV not available; cannot publish RTSP frames")
			return

		while self._running:
			try:
				# STEP 1: Connect to RTSP stream
				print(f"[publisher {self.camera_id}] 🎥 Connecting to RTSP: {self.source_uri}")
				rtsp_container = av.open(
					self.source_uri,
					format="rtsp",
					options={
						"rtsp_transport": "tcp",  # Use TCP for reliability
						"max_delay": "0",  # Low latency
					},
				)

				# STEP 2: Get video stream
				video_stream = rtsp_container.streams.video[0]
				
				# Try to get camera's native FPS from stream metadata
				try:
					camera_fps = float(video_stream.average_rate)
					# print(f"[publisher {self.camera_id}] 📊 Camera native FPS: {camera_fps:.1f}")
				except (AttributeError, ValueError, TypeError):
					camera_fps = None
					print(f"[publisher {self.camera_id}] 📊 Camera FPS: unknown (will measure)")

				# STEP 3: Decode and publish frames
				frame_index = 0
				stat_last = _now_monotonic()  # For FPS statistics
				stat_frames = 0
				last_frame_time = _now_monotonic()  # For measuring actual FPS

				# Loop through frames - decode as fast as camera provides
				for frame in rtsp_container.decode(video_stream):
					# Check if we should stop
					if not self._running:
						break

					# Check for stop command (non-blocking)
					self._check_stop_command()

					# STEP 4: Convert frame to BGR numpy array
					frame_bgr = frame.to_ndarray(format="bgr24")
					height, width = int(frame_bgr.shape[0]), int(frame_bgr.shape[1])
					frame_index += 1

					# STEP 5: Calculate actual FPS from frame timing
					now_ts = _now_monotonic()
					if frame_index > 1:
						frame_delta = now_ts - last_frame_time
						actual_fps = 1.0 / frame_delta if frame_delta > 0 else 0
					else:
						actual_fps = 0
					last_frame_time = now_ts

					# STEP 6: Store frame in shared memory (overwrites previous)
					payload = {
						"shape": (height, width, 3),  # Frame dimensions
						"dtype": "uint8",  # Data type
						"frame_index": frame_index,  # Sequential frame number
						"ts_monotonic": now_ts,  # Timestamp
						"camera_fps": camera_fps,  # From stream metadata (if available)
						"actual_fps": actual_fps,  # Measured from frame timing
						"bytes": frame_bgr.tobytes(),  # Frame data as bytes
					}
					self.shared_store[self.camera_id] = payload

					# Print statistics every second
					stat_frames += 1
					if (now_ts - stat_last) >= 1.0:
						print(f"[publisher {self.camera_id}] ⏱️  {stat_frames} frames/s (actual: {actual_fps:.1f} fps)")
						stat_last = now_ts
						stat_frames = 0

				# If loop exits (connection lost), wait and reconnect
				print(f"[publisher {self.camera_id}] ⚠️  Connection lost, reconnecting in {self.reconnect_delay}s...")
				time.sleep(self.reconnect_delay)

			except Exception as exc:  # noqa: BLE001
				# On any error, log and reconnect
				self._publish_error(f"RTSP error: {exc}")
				print(f"[publisher {self.camera_id}] ⚠️  Error occurred, reconnecting in {self.reconnect_delay}s...")
				time.sleep(self.reconnect_delay)

	def _check_stop_command(self) -> None:
		"""
		Check for stop command (non-blocking).
		
		This allows graceful shutdown when runner wants to stop the publisher.
		"""
		try:
			while True:
				command: CameraCommand = self.command_queue.get_nowait()
				if command.kind == "stop":
					print(f"[publisher {self.camera_id}] 🛑 Stop command received")
					self._running = False
					break
		except Exception:
			# Queue is empty, no commands to process
			pass

	def _publish_error(self, message: str) -> None:
		"""
		Store error message in shared_store so consumers know something went wrong.
		
		This allows agents/live stream to detect when camera is unavailable.
		"""
		self.shared_store[self.camera_id] = {
			"error": str(message),
			"ts_monotonic": _now_monotonic(),
		}




