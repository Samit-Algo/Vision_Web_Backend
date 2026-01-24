"""
Sources Module
--------------

Provides frame acquisition from RTSP cameras:
- HubSource: Reads frames from shared memory (RTSP cameras)
- SourceFactory: Creates HubSource based on task config
"""

from app.processing.sources.hub_source import HubSource
from app.processing.sources.source_factory import create_source, Source

__all__ = ["HubSource", "create_source", "Source"]
