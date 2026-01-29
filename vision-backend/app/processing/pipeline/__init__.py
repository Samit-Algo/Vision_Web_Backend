"""
Pipeline Module
---------------

Orchestrates the processing pipeline stages:
- PipelineRunner: Main orchestration loop
- PipelineContext: Task configuration and state
- Stages: Individual processing stages
"""

from app.processing.pipeline.context import PipelineContext
from app.processing.pipeline.runner import PipelineRunner

__all__ = ["PipelineRunner", "PipelineContext"]
