"""
Auto-ARIMA Model for Time Series Forecasting

Uses pmdarima for automatic ARIMA order selection.
Includes SARIMA for seasonal patterns.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
import warnings
import pickle
import os

from .base_model import StatisticalModel

# Try to import pmdarima
try:
    import pmdarima as pm
    from pmdarima import auto_arima
    PMDARIMA_AVAILABLE = True
except ImportError:
    PMDARIMA_AVAILABLE = False
    warnings.warn("pmdarima not installed. Install with: pip install pmdarima")


class ARIMAModel(StatisticalModel):
    """
    Auto-ARIMA Model for time series forecasting.
    
    Automatically selects the best (p, d, q) order based on AIC/BIC.
    Supports seasonal ARIMA (SARIMA) with (P, D, Q, m) parameters.
    """
    
    def __init__(
        self,
        seasonal: bool = True,
        m: int = 7,  # Weekly seasonality
        max_p: int = 5,
        max_q: int = 5,
        max_d: int = 2,
        max_P: int = 2,
        max_Q: int = 2,
        max_D: int = 1,
        information_criterion: str = 'aic',
        stepwise: bool = True,
        suppress_warnings: bool = True,
        **kwargs
    ):
        """
        Initialize ARIMA model.
        
        Args:
            seasonal: Whether to fit seasonal ARIMA
            m: Seasonal period (7 for weekly, 12 for monthly, 365 for yearly)
            max_p, max_q, max_d: Maximum orders for non-seasonal components
            max_P, max_Q, max_D: Maximum orders for seasonal components
            information_criterion: 'aic', 'bic', or 'hqic'
            stepwise: Use stepwise algorithm for faster selection
        """
        super().__init__(name="Auto-ARIMA", **kwargs)
        
        if not PMDARIMA_AVAILABLE:
            raise ImportError("pmdarima required. Install with: pip install pmdarima")
        
        self.seasonal = seasonal
        self.m = m if seasonal else 1
        self.max_p = max_p
        self.max_q = max_q
        self.max_d = max_d
        self.max_P = max_P
        self.max_Q = max_Q
        self.max_D = max_D
        self.information_criterion = information_criterion
        self.stepwise = stepwise
        self.suppress_warnings = suppress_warnings
        
        self.order = None
        self.seasonal_order = None
        
    def build(self) -> None:
        """ARIMA model is built during fit."""
        pass
    
    def fit(
        self,
        y_train: np.ndarray,
        X_train: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Fit Auto-ARIMA model.
        
        Args:
            y_train: Target time series (1D array)
            X_train: Exogenous variables (optional)
            
        Returns:
            Dictionary with model info and metrics
        """
        # Ensure 1D
        y = np.array(y_train).flatten()
        X = X_train if X_train is not None else None
        
        with warnings.catch_warnings():
            if self.suppress_warnings:
                warnings.simplefilter("ignore")
            
            self.model = auto_arima(
                y,
                X=X,
                seasonal=self.seasonal,
                m=self.m,
                max_p=self.max_p,
                max_q=self.max_q,
                max_d=self.max_d,
                max_P=self.max_P,
                max_Q=self.max_Q,
                max_D=self.max_D,
                information_criterion=self.information_criterion,
                stepwise=self.stepwise,
                suppress_warnings=self.suppress_warnings,
                error_action='ignore',
                trace=False
            )
        
        self.is_fitted = True
        self.order = self.model.order
        self.seasonal_order = self.model.seasonal_order
        
        return {
            'order': self.order,
            'seasonal_order': self.seasonal_order,
            'aic': self.model.aic(),
            'bic': self.model.bic()
        }
    
    def predict(self, n_periods: int = 1, X: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            n_periods: Number of periods to forecast
            X: Exogenous variables for forecast period
            
        Returns:
            Predicted values
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        predictions = self.model.predict(n_periods=n_periods, X=X)
        return predictions
    
    def predict_with_confidence(
        self,
        n_periods: int = 1,
        alpha: float = 0.05,
        X: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict with confidence intervals.
        
        Returns:
            (predictions, lower_bound, upper_bound)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        predictions, conf_int = self.model.predict(
            n_periods=n_periods,
            X=X,
            return_conf_int=True,
            alpha=alpha
        )
        
        return predictions, conf_int[:, 0], conf_int[:, 1]
    
    def save(self, path: str) -> None:
        """Save model to disk."""
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
    
    def load(self, path: str) -> None:
        """Load model from disk."""
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        self.is_fitted = True
        self.order = self.model.order
        self.seasonal_order = self.model.seasonal_order
    
    def summary(self) -> str:
        """Get model summary."""
        if self.is_fitted:
            return f"ARIMA{self.order} x {self.seasonal_order}"
        return "Auto-ARIMA (not fitted)"


def create_arima_model(**kwargs) -> ARIMAModel:
    """Factory function to create ARIMA model."""
    return ARIMAModel(**kwargs)
