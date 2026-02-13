"""
Sources Module
--------------

Provides frame acquisition:
- HubSource: Reads frames from shared memory (RTSP cameras)
- VideoFileSource: Reads frames from a static video file
- SourceFactory: Creates source from task (file or RTSP)
"""

from .hub_source import HubSource
from .source_factory import Source, create_source
from .video_file_source import VideoFileSource

__all__ = ["HubSource", "VideoFileSource", "create_source", "Source"]
