"""
Data input (frame sources)
--------------------------

Provides frame acquisition for the pipeline:
- HubSource: reads from shared memory (RTSP cameras, published by CameraPublisher)
- VideoFileSource: reads from a static video file
- create_source(): builds the right source from task config
"""

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
from .hub_source import HubSource
from .source_factory import Source, create_source
from .video_file_source import VideoFileSource

__all__ = ["HubSource", "VideoFileSource", "create_source", "Source"]
