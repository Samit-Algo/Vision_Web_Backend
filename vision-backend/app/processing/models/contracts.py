"""
Model Data Contracts
--------------------

Defines the interfaces and contracts for model providers and model results.
These contracts are frozen - they define the stable API between models and the pipeline.
"""

from typing import Protocol, Any, Optional
import numpy as np


class Model(Protocol):
    """
    Protocol defining what a loaded model instance looks like.
    
    Models must support inference via callable interface: model(frame) -> results
    """
    
    def __call__(self, frame: np.ndarray, verbose: bool = False) -> Any:
        """
        Run inference on a frame.
        
        Args:
            frame: Input frame as numpy array (BGR format)
            verbose: Whether to print verbose output
        
        Returns:
            Model results (format depends on model type, e.g., ultralytics Results object)
        """
        ...


class ModelResult(Protocol):
    """
    Protocol defining the structure of model inference results.
    
    This is intentionally minimal - we preserve the original model output format
    (e.g., ultralytics Results object) to maintain compatibility.
    """
    pass


class Provider(Protocol):
    """
    Protocol defining the interface that model providers must implement.
    
    Providers are responsible for loading specific model types.
    """
    
    def load(self, model_id: str) -> Optional[Model]:
        """
        Load a model instance.
        
        Args:
            model_id: Model identifier (path or name)
        
        Returns:
            Loaded model instance or None if loading failed
        """
        ...
