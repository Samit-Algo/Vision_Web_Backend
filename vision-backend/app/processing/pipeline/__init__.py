"""
Pipeline Module
---------------

Orchestrates the processing pipeline stages:
- PipelineRunner: Main orchestration loop
- PipelineContext: Task configuration and state
- Stages: Individual processing stages
"""

from app.processing.pipeline.runner import PipelineRunner
from app.processing.pipeline.context import PipelineContext

__all__ = ["PipelineRunner", "PipelineContext"]
