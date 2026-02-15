"""
Event Session Manager
=====================

Manages event sessions with runtime video splitting.
- Creates sessions when events are detected
- Collects frames during active events
- Splits videos at runtime (every 5 minutes)
- Saves videos to local files (no Kafka for videos)
- Updates event documents in MongoDB with video_path when session closes
"""
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Optional, Any, List
from queue import Queue
import numpy as np

from .datetime_utils import now
from .video_encoder import (
    encode_frames_to_file,
    get_video_duration_seconds
)
from .event_notifier import send_event_to_backend_sync
from .db import get_collection
from ..domain.constants.event_fields import EventFields
from ..core.config import get_settings


class SessionState(str, Enum):
    """Session state machine states."""
    ACTIVE = "ACTIVE"  # Collecting frames
    ENCODING = "ENCODING"  # Currently encoding a chunk
    CLOSING = "CLOSING"  # Session is being closed


@dataclass
class EventSession:
    """Represents an active event session."""
    session_id: str
    agent_id: str
    rule_index: int
    event_label: str
    camera_id: Optional[str]
    agent_name: str
    event_type: str = ""  # e.g. "fall_detected" for UI to show red alert
    
    # State management
    state: SessionState = SessionState.ACTIVE
    start_time: datetime = field(default_factory=now)
    last_event_time: datetime = field(default_factory=now)
    chunk_start_time: datetime = field(default_factory=now)
    chunk_number: int = 0
    
    # Frame storage (bounded deque)
    frames: deque = field(default_factory=lambda: deque())
    frame_timestamps: deque = field(default_factory=lambda: deque())
    
    # Metadata
    detections_snapshot: Optional[Dict[str, Any]] = None
    video_timestamp: Optional[str] = None
    
    # Video paths (track saved video files)
    video_paths: List[str] = field(default_factory=list)
    
    # Configuration (set when session is created)
    fps: int = 5
    chunk_duration_seconds: int = 300  # 5 minutes
    chunk_frame_limit: int = 0  # Will be calculated
    
    def __post_init__(self):
        """Calculate chunk frame limit based on FPS and duration."""
        self.chunk_frame_limit = self.fps * self.chunk_duration_seconds
        # Initialize frames deque with maxlen to prevent unbounded growth
        if not isinstance(self.frames, deque) or self.frames.maxlen is None:
            # Add some buffer (10% extra) to handle FPS variations
            maxlen = int(self.chunk_frame_limit * 1.1)
            self.frames = deque(maxlen=maxlen)
            self.frame_timestamps = deque(maxlen=maxlen)


class EventSessionManager:
    """
    Manages event sessions with runtime video splitting.
    
    Thread-safe session manager that:
    - Handles event frames from agents
    - Splits videos at runtime (every 5 minutes)
    - Manages session lifecycle
    - Saves videos to local files
    - Updates event documents in MongoDB with video_path when session closes
    """
    
    def __init__(self):
        """Initialize the session manager."""
        self.sessions: Dict[str, EventSession] = {}
        self.lock = threading.Lock()
        self.running = False
        self.background_thread: Optional[threading.Thread] = None
        self.encode_queue: Queue = Queue()
        self.encode_worker_thread: Optional[threading.Thread] = None
        
        settings = get_settings()
        self.session_timeout_seconds = getattr(
            settings, 'event_session_timeout_seconds', 30
        )
        self.check_interval_seconds = getattr(
            settings, 'event_session_check_interval_seconds', 5
        )
        self.chunk_duration_seconds = getattr(
            settings, 'event_video_chunk_duration_seconds', 300
        )
        self.video_fps = getattr(settings, 'event_video_fps', 5)
        self.video_width = getattr(settings, 'event_video_resolution_width', 1280)
        self.video_height = getattr(settings, 'event_video_resolution_height', 720)
        self.video_save_directory = getattr(settings, 'event_video_save_directory', './event_videos')
        self.video_save_enabled = getattr(settings, 'event_video_save_enabled', True)
        
        # Convert relative path to absolute
        if not os.path.isabs(self.video_save_directory):
            self.video_save_directory = os.path.abspath(self.video_save_directory)
        
        # Ensure directory exists
        if self.video_save_enabled:
            try:
                os.makedirs(self.video_save_directory, exist_ok=True)
                print(
                    f"[EventSessionManager] 📁 Video save directory created/verified: {self.video_save_directory}"
                )
            except Exception as e:
                print(f"[EventSessionManager] ⚠️  Failed to create video save directory: {e}")
        
        # Log configuration
        print(
            f"[EventSessionManager] 📁 Video save config: "
            f"enabled={self.video_save_enabled} | directory={self.video_save_directory}"
        )
    
    def start(self):
        """Start background workers."""
        if self.running:
            return
        
        self.running = True
        
        # Start session expiration checker
        self.background_thread = threading.Thread(
            target=self._background_worker,
            daemon=True,
            name="EventSessionChecker"
        )
        self.background_thread.start()
        
        # Start video encoding worker
        self.encode_worker_thread = threading.Thread(
            target=self._encode_worker,
            daemon=True,
            name="VideoEncoderWorker"
        )
        self.encode_worker_thread.start()
        
        print("[EventSessionManager] ✅ Started background workers")
    
    def stop(self):
        """Stop background workers and flush all sessions."""
        self.running = False
        
        # Wait for background thread
        if self.background_thread and self.background_thread.is_alive():
            self.background_thread.join(timeout=5.0)
        
        # Flush all active sessions
        with self.lock:
            session_keys = list(self.sessions.keys())
            for session_key in session_keys:
                session = self.sessions.get(session_key)
                if session and session.state == SessionState.ACTIVE:
                    self._close_session(session, flush_remaining=True)
        
        # Wait for encoding queue to drain
        self.encode_queue.join()
        
        # Wait for encode worker
        if self.encode_worker_thread and self.encode_worker_thread.is_alive():
            self.encode_worker_thread.join(timeout=10.0)
        
        print("[EventSessionManager] ⏹️  Stopped")
    
    def handle_event_frame(
        self,
        agent_id: str,
        rule_index: int,
        event_label: str,
        frame: np.ndarray,
        camera_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        detections: Optional[Dict[str, Any]] = None,
        video_timestamp: Optional[str] = None,
        fps: Optional[int] = None,
        event_type: Optional[str] = None,
    ) -> None:
        """
        Handle an event frame from an agent.
        
        This is the main entry point called from agent_main.py when an event is detected.
        
        Args:
            agent_id: Agent identifier
            rule_index: Rule index that triggered the event
            event_label: Event label (e.g., "person without helmet")
            frame: Annotated frame (numpy array)
            camera_id: Camera identifier
            agent_name: Agent name
            detections: Detection details
            video_timestamp: Video timestamp string
            fps: Agent FPS (for video encoding)
            event_type: Event type (e.g. "fall_detected") for UI alerts (red for fall).
        """
        session_key = f"{agent_id}_{rule_index}"
        current_time = now()
        
        with self.lock:
            # Get or create session
            session = self.sessions.get(session_key)
            
            if session is None:
                # Create new session
                session = EventSession(
                    session_id=f"{agent_id}_{rule_index}_{int(current_time.timestamp())}",
                    agent_id=agent_id,
                    rule_index=rule_index,
                    event_label=event_label,
                    camera_id=camera_id,
                    agent_name=agent_name or agent_id,
                    event_type=(event_type or "").strip(),
                    fps=fps or self.video_fps,
                    chunk_duration_seconds=self.chunk_duration_seconds,
                    detections_snapshot=detections,
                    video_timestamp=video_timestamp
                )
                self.sessions[session_key] = session
                print(
                    f"[EventSessionManager] 🆕 Created session: {session.session_id} "
                    f"| agent={agent_name} | rule={rule_index} | event={event_label}"
                )
                
                # Send immediate notification (first event) - DB + Kafka
                self._send_immediate_notification(session, frame, detections, video_timestamp)
            
            # Check if session is in valid state
            if session.state not in [SessionState.ACTIVE, SessionState.ENCODING]:
                # Session is closing, skip this frame
                return
            
            # Check if we need to split (runtime check)
            # Use both frame count and wall-clock time for robustness
            frame_count = len(session.frames)
            chunk_duration = (current_time - session.chunk_start_time).total_seconds()
            
            should_split = (
                frame_count >= session.chunk_frame_limit or
                chunk_duration >= session.chunk_duration_seconds
            )
            
            if should_split and session.state == SessionState.ACTIVE and frame_count > 0:
                # Runtime split needed
                print(
                    f"[EventSessionManager] ✂️  Runtime split: session={session.session_id} "
                    f"| chunk={session.chunk_number} | frames={frame_count} | duration={chunk_duration:.1f}s"
                )
                self._enqueue_chunk_encode(session, is_final=False)
                # Reset for next chunk
                session.frames.clear()
                session.frame_timestamps.clear()
                session.chunk_start_time = current_time
                session.chunk_number += 1
            
            # Add frame to current chunk
            session.frames.append(frame.copy())
            session.frame_timestamps.append(current_time)
            session.last_event_time = current_time
            
            # Update metadata if provided
            if detections:
                session.detections_snapshot = detections
            if video_timestamp:
                session.video_timestamp = video_timestamp
    
    def _send_immediate_notification(
        self,
        session: EventSession,
        frame: np.ndarray,
        detections: Optional[Dict[str, Any]],
        video_timestamp: Optional[str]
    ) -> None:
        """Send immediate notification on first event (DB + Kafka)."""
        try:
            event = {
                "label": session.event_label,
                "rule_index": session.rule_index,
                "event_type": session.event_type or "",
            }
            
            # Send event notification (dual write: DB + Kafka)
            send_event_to_backend_sync(
                event=event,
                annotated_frame=frame,
                agent_id=session.agent_id,
                agent_name=session.agent_name,
                camera_id=session.camera_id,
                video_timestamp=video_timestamp,
                detections=detections,
                session_id=session.session_id
            )
            print(
                f"[EventSessionManager] ✅ Immediate notification sent: "
                f"session={session.session_id} | event={session.event_label}"
            )
        except Exception as e:
            print(f"[EventSessionManager] ❌ Error sending immediate notification: {e}")
            import traceback
            print(f"[EventSessionManager] Traceback: {traceback.format_exc()}")
    
    def _enqueue_chunk_encode(self, session: EventSession, is_final: bool) -> None:
        """Enqueue a chunk for encoding (non-blocking)."""
        if session.state == SessionState.ENCODING:
            # Already encoding, skip
            return
        
        session.state = SessionState.ENCODING
        
        # Create encode job
        encode_job = {
            "session": session,
            "frames": list(session.frames),  # Copy frames list
            "frame_timestamps": list(session.frame_timestamps),  # Copy timestamps
            "is_final": is_final,
            "chunk_number": session.chunk_number
        }
        
        self.encode_queue.put(encode_job)
        print(
            f"[EventSessionManager] 📥 Enqueued encode job: "
            f"session={session.session_id} | chunk={session.chunk_number} | final={is_final}"
        )
    
    def _encode_worker(self):
        """Background worker that encodes video chunks and saves to files."""
        print("[EventSessionManager] 🎬 Encode worker started")
        
        while self.running:
            try:
                # Get job from queue (with timeout to allow checking self.running)
                try:
                    encode_job = self.encode_queue.get(timeout=1.0)
                except:
                    continue
                
                session = encode_job["session"]
                frames = encode_job["frames"]
                is_final = encode_job["is_final"]
                chunk_number = encode_job["chunk_number"]
                
                if not frames:
                    # No frames to encode
                    self.encode_queue.task_done()
                    with self.lock:
                        if session.state == SessionState.ENCODING:
                            session.state = SessionState.ACTIVE
                    continue
                
                print(
                    f"[EventSessionManager] 🎬 Encoding: session={session.session_id} "
                    f"| chunk={chunk_number} | frames={len(frames)} | final={is_final}"
                )
                
                # Calculate chunk duration
                chunk_duration = get_video_duration_seconds(len(frames), session.fps)
                chunk_start_time = session.chunk_start_time
                
                # Save video to local file
                saved_file_path = None
                if self.video_save_enabled:
                    try:
                        # Create meaningful filename
                        timestamp_str = chunk_start_time.strftime("%Y%m%d_%H%M%S")
                        filename = (
                            f"{session.session_id}_chunk{chunk_number:03d}_"
                            f"{timestamp_str}_{'final' if is_final else 'partial'}.mp4"
                        )
                        # Sanitize filename (remove invalid characters)
                        filename = "".join(c for c in filename if c.isalnum() or c in ('_', '-', '.'))
                        output_path = os.path.join(self.video_save_directory, filename)
                        
                        print(
                            f"[EventSessionManager] 💾 Saving video: "
                            f"path={output_path} | frames={len(frames)}"
                        )
                        
                        # Use H.264 codec for browser compatibility (Chromium supports H.264)
                        # Try 'avc1' first (H.264), fallback to 'H264' if needed
                        saved_file_path = None
                        for codec_attempt in ["avc1", "H264", "mp4v"]:
                            try:
                                saved_file_path = encode_frames_to_file(
                                    frames=frames,
                                    output_path=output_path,
                                    fps=session.fps,
                                    width=self.video_width,
                                    height=self.video_height,
                                    codec=codec_attempt,
                                    output_format="mp4"
                                )
                                if saved_file_path:
                                    print(
                                        f"[EventSessionManager] ✅ Video encoded with codec: {codec_attempt}"
                                    )
                                    break
                            except Exception as e:
                                print(
                                    f"[EventSessionManager] ⚠️  Codec {codec_attempt} failed: {e}, trying next..."
                                )
                                continue
                        
                        if not saved_file_path:
                            print(
                                f"[EventSessionManager] ❌ All codec attempts failed for video: {output_path}"
                            )
                        
                        if saved_file_path:
                            # Track video path in session
                            session.video_paths.append(saved_file_path)
                            print(
                                f"[EventSessionManager] ✅ Video saved: {saved_file_path}"
                            )
                        else:
                            print(
                                f"[EventSessionManager] ⚠️  Video encoding returned None"
                            )
                    except Exception as e:
                        print(f"[EventSessionManager] ⚠️  Error saving video: {e}")
                        import traceback
                        print(f"[EventSessionManager] Traceback: {traceback.format_exc()}")
                else:
                    print(
                        f"[EventSessionManager] ⚠️  Video save is DISABLED"
                    )
                
                # Mark job as done
                self.encode_queue.task_done()
                
                # Update session state and MongoDB if final chunk
                with self.lock:
                    if session.state == SessionState.ENCODING:
                        if is_final:
                            session.state = SessionState.CLOSING
                            # Update event document in MongoDB with video path
                            self._update_event_with_video_path(session)
                        else:
                            session.state = SessionState.ACTIVE
                
            except Exception as e:
                print(f"[EventSessionManager] ⚠️  Error in encode worker: {e}")
                import traceback
                print(f"[EventSessionManager] Traceback: {traceback.format_exc()}")
                try:
                    self.encode_queue.task_done()
                except:
                    pass
        
        print("[EventSessionManager] 🎬 Encode worker stopped")
    
    def _update_event_with_video_path(self, session: EventSession) -> None:
        """
        Update event document in MongoDB with video_path when session closes.
        
        Finds the event by session_id and updates metadata with video_path.
        """
        if not session.video_paths:
            print(
                f"[EventSessionManager] ⚠️  No video paths to update for session: {session.session_id}"
            )
            return
        
        try:
            events_collection = get_collection("events")
            
            # Use the primary video path (final chunk - most recent)
            video_path = session.video_paths[-1]
            
            # Update event document - add video_path to metadata
            # Find event by session_id
            result = events_collection.update_many(
                {EventFields.SESSION_ID: session.session_id},
                {
                    "$set": {
                        f"{EventFields.METADATA}.video_path": video_path,
                        f"{EventFields.METADATA}.video_paths": session.video_paths,  # Store all paths if multiple chunks
                    }
                }
            )
            
            if result.modified_count > 0:
                print(
                    f"[EventSessionManager] ✅ Updated event document with video_path: "
                    f"session={session.session_id} | video_path={video_path} | "
                    f"updated_count={result.modified_count}"
                )
            else:
                print(
                    f"[EventSessionManager] ⚠️  No event document found to update: "
                    f"session={session.session_id}"
                )
        except Exception as e:
            print(f"[EventSessionManager] ❌ Error updating event with video_path: {e}")
            import traceback
            print(f"[EventSessionManager] Traceback: {traceback.format_exc()}")
    
    def _background_worker(self):
        """Background worker that checks for expired sessions."""
        print("[EventSessionManager] 🔍 Background worker started")
        
        while self.running:
            try:
                time.sleep(self.check_interval_seconds)
                
                if not self.running:
                    break
                
                current_time = now()
                expired_sessions = []
                
                with self.lock:
                    for session_key, session in list(self.sessions.items()):
                        if session.state not in [SessionState.ACTIVE, SessionState.ENCODING]:
                            continue
                        
                        # Check if session expired (no event for timeout period)
                        time_since_last_event = (current_time - session.last_event_time).total_seconds()
                        
                        if time_since_last_event >= self.session_timeout_seconds:
                            expired_sessions.append(session_key)
                
                # Close expired sessions (outside lock to avoid deadlock)
                for session_key in expired_sessions:
                    with self.lock:
                        session = self.sessions.get(session_key)
                        if session:
                            self._close_session(session, flush_remaining=True)
                            del self.sessions[session_key]
            
            except Exception as e:
                print(f"[EventSessionManager] ⚠️  Error in background worker: {e}")
                import traceback
                print(f"[EventSessionManager] Traceback: {traceback.format_exc()}")
        
        print("[EventSessionManager] 🔍 Background worker stopped")
    
    def _close_session(self, session: EventSession, flush_remaining: bool = False) -> None:
        """Close a session and encode remaining frames if any."""
        if session.state == SessionState.CLOSING:
            return
        
        print(
            f"[EventSessionManager] 🔒 Closing session: {session.session_id} "
            f"| chunks={session.chunk_number} | remaining_frames={len(session.frames)}"
        )
        
        if flush_remaining and len(session.frames) > 0:
            # Encode remaining frames as final chunk
            self._enqueue_chunk_encode(session, is_final=True)
        else:
            session.state = SessionState.CLOSING
            # Update event document even if no remaining frames
            if session.video_paths:
                self._update_event_with_video_path(session)


# Global singleton instance
_session_manager: Optional[EventSessionManager] = None


def get_event_session_manager() -> EventSessionManager:
    """Get the global EventSessionManager instance (singleton)."""
    global _session_manager
    if _session_manager is None:
        _session_manager = EventSessionManager()
        _session_manager.start()
    return _session_manager
