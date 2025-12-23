"""
Exponential Smoothing Models for Time Series Forecasting

Includes:
- Simple Exponential Smoothing
- Holt's Linear Trend
- Holt-Winters (Triple Exponential Smoothing)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
import warnings
import pickle

from .base_model import StatisticalModel

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
    from statsmodels.tsa.statespace.exponential_smoothing import ExponentialSmoothing as ETSModel
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    warnings.warn("statsmodels not installed. Install with: pip install statsmodels")


class ExponentialSmoothingModel(StatisticalModel):
    """
    Holt-Winters Exponential Smoothing Model.
    
    Handles trend and seasonal components in time series.
    """
    
    def __init__(
        self,
        trend: str = 'add',  # 'add', 'mul', or None
        seasonal: str = 'add',  # 'add', 'mul', or None
        seasonal_periods: int = 7,  # Weekly seasonality
        damped_trend: bool = False,
        use_boxcox: bool = False,
        **kwargs
    ):
        """
        Initialize Exponential Smoothing model.
        
        Args:
            trend: Type of trend component ('add', 'mul', or None)
            seasonal: Type of seasonal component ('add', 'mul', or None)
            seasonal_periods: Number of periods in a complete seasonal cycle
            damped_trend: Whether to damp the trend
            use_boxcox: Apply Box-Cox transformation
        """
        super().__init__(name="Exponential Smoothing", **kwargs)
        
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels required. Install with: pip install statsmodels")
        
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.damped_trend = damped_trend
        self.use_boxcox = use_boxcox
        
    def build(self) -> None:
        """Model is built during fit."""
        pass
    
    def fit(
        self,
        y_train: np.ndarray,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Fit Exponential Smoothing model.
        
        Args:
            y_train: Target time series (1D array)
            
        Returns:
            Dictionary with model info
        """
        y = np.array(y_train).flatten()
        
        # Handle very short series
        if len(y) < self.seasonal_periods * 2:
            self.seasonal = None
            warnings.warn("Series too short for seasonality, using non-seasonal model")
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            model = ExponentialSmoothing(
                y,
                trend=self.trend,
                seasonal=self.seasonal,
                seasonal_periods=self.seasonal_periods if self.seasonal else None,
                damped_trend=self.damped_trend,
                use_boxcox=self.use_boxcox
            )
            
            self.model = model.fit(optimized=True)
        
        self.is_fitted = True
        
        return {
            'aic': self.model.aic if hasattr(self.model, 'aic') else None,
            'bic': self.model.bic if hasattr(self.model, 'bic') else None
        }
    
    def predict(self, n_periods: int = 1) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            n_periods: Number of periods to forecast
            
        Returns:
            Predicted values
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        predictions = self.model.forecast(n_periods)
        return np.array(predictions)
    
    def predict_with_confidence(
        self,
        n_periods: int = 1,
        alpha: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict with confidence intervals.
        
        Returns:
            (predictions, lower_bound, upper_bound)
        """
        predictions = self.predict(n_periods)
        
        # Approximate confidence intervals using residual std
        residuals = self.model.resid
        std = np.std(residuals)
        z = 1.96  # 95% CI
        
        lower = predictions - z * std
        upper = predictions + z * std
        
        return predictions, lower, upper
    
    def save(self, path: str) -> None:
        """Save model to disk."""
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
    
    def load(self, path: str) -> None:
        """Load model from disk."""
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        self.is_fitted = True
    
    def summary(self) -> str:
        """Get model summary."""
        if self.is_fitted:
            return f"ETS(trend={self.trend}, seasonal={self.seasonal}, periods={self.seasonal_periods})"
        return "Exponential Smoothing (not fitted)"


def create_exponential_smoothing_model(**kwargs) -> ExponentialSmoothingModel:
    """Factory function to create Exponential Smoothing model."""
    return ExponentialSmoothingModel(**kwargs)
