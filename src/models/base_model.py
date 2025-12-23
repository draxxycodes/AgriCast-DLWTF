"""
Base Model Abstract Class

All models inherit from this base class for consistent interface.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd


class BaseModel(ABC):
    """Abstract base class for all forecasting models."""
    
    def __init__(self, name: str, **kwargs):
        self.name = name
        self.model = None
        self.is_fitted = False
        self.config = kwargs
        
    @abstractmethod
    def build(self) -> None:
        """Build the model architecture."""
        pass
    
    @abstractmethod
    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs) -> Dict[str, Any]:
        """
        Train the model.
        
        Returns:
            Dictionary with training history/metrics
        """
        pass
    
    @abstractmethod
    def predict(self, X) -> np.ndarray:
        """Make predictions."""
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save model to disk."""
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """Load model from disk."""
        pass
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return self.config
    
    def summary(self) -> str:
        """Get model summary."""
        return f"{self.name} - Fitted: {self.is_fitted}"


class StatisticalModel(BaseModel):
    """Base class for statistical time series models."""
    
    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
        self.requires_1d = True  # Statistical models typically need 1D input
        

class DeepLearningModel(BaseModel):
    """Base class for deep learning models."""
    
    def __init__(self, name: str, sequence_length: int, n_features: int, **kwargs):
        super().__init__(name, **kwargs)
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.requires_1d = False
        
    def count_params(self) -> int:
        """Count trainable parameters."""
        if self.model is not None:
            return self.model.count_params()
        return 0
