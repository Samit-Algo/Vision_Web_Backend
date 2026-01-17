"""
Video Encoder
=============

Utility for encoding frames to video files.
Handles frame-to-video conversion with proper FPS and resolution settings.
"""
import base64
import io
import os
from pathlib import Path
from typing import List, Optional, Tuple
from datetime import datetime
import numpy as np

try:
    import cv2  # type: ignore
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None


def encode_frames_to_video(
    frames: List[np.ndarray],
    fps: int,
    width: Optional[int] = None,
    height: Optional[int] = None,
    codec: str = "mp4v",
    output_format: str = "mp4"
) -> Optional[bytes]:
    """
    Encode a list of frames into a video file in memory.
    
    Args:
        frames: List of numpy arrays (frames) in BGR format
        fps: Frames per second for the output video
        width: Output video width (defaults to frame width)
        height: Output video height (defaults to frame height)
        codec: Video codec (default: 'mp4v', alternatives: 'H264', 'XVID')
        output_format: Output format extension (default: 'mp4')
    
    Returns:
        Video file bytes, or None if encoding fails
    """
    if not CV2_AVAILABLE:
        print("[video_encoder] ⚠️  OpenCV not available, cannot encode video")
        return None
    
    if not frames:
        print("[video_encoder] ⚠️  No frames to encode")
        return None
    
    try:
        # Get dimensions from first frame
        first_frame = frames[0]
        if len(first_frame.shape) != 3:
            print(f"[video_encoder] ⚠️  Invalid frame shape: {first_frame.shape}")
            return None
        
        frame_height, frame_width = first_frame.shape[:2]
        output_width = width if width is not None else frame_width
        output_height = height if height is not None else frame_height
        
        # Create video writer in memory
        fourcc = cv2.VideoWriter_fourcc(*codec)
        
        # Use BytesIO as a file-like object
        # Note: OpenCV VideoWriter doesn't directly support BytesIO, so we'll use a temp approach
        # For production, consider using tempfile and reading back, or use ffmpeg-python
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{output_format}") as tmp_file:
            temp_path = tmp_file.name
        
        try:
            video_writer = cv2.VideoWriter(
                temp_path,
                fourcc,
                float(fps),
                (output_width, output_height)
            )
            
            if not video_writer.isOpened():
                print(f"[video_encoder] ⚠️  Failed to open video writer with codec {codec}")
                return None
            
            # Write all frames
            for frame in frames:
                # Resize if needed
                if frame.shape[:2] != (output_height, output_width):
                    resized_frame = cv2.resize(frame, (output_width, output_height))
                else:
                    resized_frame = frame
                
                video_writer.write(resized_frame)
            
            video_writer.release()
            
            # Read the video file back into memory
            with open(temp_path, 'rb') as f:
                video_bytes = f.read()
            
            return video_bytes
            
        finally:
            # Clean up temp file
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            except Exception:
                pass
                
    except Exception as e:
        print(f"[video_encoder] ⚠️  Error encoding video: {e}")
        import traceback
        print(f"[video_encoder] Traceback: {traceback.format_exc()}")
        return None


def encode_frames_to_base64_video(
    frames: List[np.ndarray],
    fps: int,
    width: Optional[int] = None,
    height: Optional[int] = None,
    codec: str = "mp4v",
    output_format: str = "mp4"
) -> Optional[str]:
    """
    Encode frames to video and return as base64 string.
    
    Args:
        frames: List of numpy arrays (frames) in BGR format
        fps: Frames per second for the output video
        width: Output video width (defaults to frame width)
        height: Output video height (defaults to frame height)
        codec: Video codec
        output_format: Output format extension
    
    Returns:
        Base64-encoded video string, or None if encoding fails
    """
    video_bytes = encode_frames_to_video(
        frames=frames,
        fps=fps,
        width=width,
        height=height,
        codec=codec,
        output_format=output_format
    )
    
    if video_bytes is None:
        return None
    
    try:
        video_base64 = base64.b64encode(video_bytes).decode('utf-8')
        return video_base64
    except Exception as e:
        print(f"[video_encoder] ⚠️  Error encoding video to base64: {e}")
        return None


def encode_frames_to_file(
    frames: List[np.ndarray],
    output_path: str,
    fps: int,
    width: Optional[int] = None,
    height: Optional[int] = None,
    codec: str = "mp4v",
    output_format: str = "mp4"
) -> Optional[str]:
    """
    Encode frames to video and save to file.
    
    Args:
        frames: List of numpy arrays (frames) in BGR format
        output_path: Full path where video should be saved
        fps: Frames per second for the output video
        width: Output video width (defaults to frame width)
        height: Output video height (defaults to frame height)
        codec: Video codec
        output_format: Output format extension
    
    Returns:
        Path to saved video file, or None if encoding fails
    """
    if not CV2_AVAILABLE:
        print("[video_encoder] ⚠️  OpenCV not available, cannot encode video")
        return None
    
    if not frames:
        print("[video_encoder] ⚠️  No frames to encode")
        return None
    
    try:
        # Get dimensions from first frame
        first_frame = frames[0]
        if len(first_frame.shape) != 3:
            print(f"[video_encoder] ⚠️  Invalid frame shape: {first_frame.shape}")
            return None
        
        frame_height, frame_width = first_frame.shape[:2]
        output_width = width if width is not None else frame_width
        output_height = height if height is not None else frame_height
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*codec)
        video_writer = cv2.VideoWriter(
            output_path,
            fourcc,
            float(fps),
            (output_width, output_height)
        )
        
        if not video_writer.isOpened():
            print(f"[video_encoder] ⚠️  Failed to open video writer with codec {codec}")
            return None
        
        # Write all frames
        for frame in frames:
            # Resize if needed
            if frame.shape[:2] != (output_height, output_width):
                resized_frame = cv2.resize(frame, (output_width, output_height))
            else:
                resized_frame = frame
            
            video_writer.write(resized_frame)
        
        video_writer.release()
        
        # Verify file was created
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(
                f"[video_encoder] ✅ Video saved: {output_path} "
                f"| size={file_size / (1024*1024):.2f}MB | frames={len(frames)}"
            )
            return output_path
        else:
            print(f"[video_encoder] ⚠️  Video file not created: {output_path}")
            return None
            
    except Exception as e:
        print(f"[video_encoder] ⚠️  Error encoding video to file: {e}")
        import traceback
        print(f"[video_encoder] Traceback: {traceback.format_exc()}")
        return None


def get_video_duration_seconds(num_frames: int, fps: int) -> float:
    """
    Calculate video duration in seconds from frame count and FPS.
    
    Args:
        num_frames: Number of frames
        fps: Frames per second
    
    Returns:
        Duration in seconds
    """
    if fps <= 0:
        return 0.0
    return num_frames / float(fps)

