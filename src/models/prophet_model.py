"""
Facebook Prophet Model for Time Series Forecasting

Prophet is designed for forecasting with:
- Strong seasonal effects
- Multiple seasonality (daily, weekly, yearly)
- Holiday effects
- Trend changes
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
import warnings
import pickle
import os

from .base_model import StatisticalModel

# Try to import prophet
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    warnings.warn("prophet not installed. Install with: pip install prophet")


class ProphetModel(StatisticalModel):
    """
    Facebook Prophet for time series forecasting.
    
    Handles:
    - Multiple seasonality (daily, weekly, yearly)
    - Holiday effects
    - Missing data
    - Trend changes (changepoints)
    """
    
    def __init__(
        self,
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = False,
        seasonality_mode: str = 'multiplicative',
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        holidays_prior_scale: float = 10.0,
        interval_width: float = 0.95,
        growth: str = 'linear',  # 'linear' or 'logistic'
        **kwargs
    ):
        """
        Initialize Prophet model.
        
        Args:
            yearly_seasonality: Include yearly seasonality
            weekly_seasonality: Include weekly seasonality
            daily_seasonality: Include daily seasonality
            seasonality_mode: 'additive' or 'multiplicative'
            changepoint_prior_scale: Flexibility of trend changes
            seasonality_prior_scale: Flexibility of seasonality
            growth: 'linear' or 'logistic' growth
        """
        super().__init__(name="Prophet", **kwargs)
        
        if not PROPHET_AVAILABLE:
            raise ImportError("prophet required. Install with: pip install prophet")
        
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.seasonality_mode = seasonality_mode
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.holidays_prior_scale = holidays_prior_scale
        self.interval_width = interval_width
        self.growth = growth
        
    def build(self) -> None:
        """Build Prophet model."""
        self.model = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            seasonality_mode=self.seasonality_mode,
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            holidays_prior_scale=self.holidays_prior_scale,
            interval_width=self.interval_width,
            growth=self.growth
        )
    
    def fit(
        self,
        df: pd.DataFrame,
        date_col: str = 'date',
        target_col: str = 'price',
        **kwargs
    ) -> Dict[str, Any]:
        """
        Fit Prophet model.
        
        Args:
            df: DataFrame with date and target columns
            date_col: Name of date column
            target_col: Name of target column
            
        Returns:
            Dictionary with model info
        """
        # Prophet requires 'ds' and 'y' columns
        prophet_df = pd.DataFrame({
            'ds': pd.to_datetime(df[date_col]),
            'y': df[target_col].values
        })
        
        self.build()
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(prophet_df)
        
        self.is_fitted = True
        
        return {
            'n_samples': len(prophet_df),
            'date_range': f"{prophet_df['ds'].min()} to {prophet_df['ds'].max()}"
        }
    
    def predict(
        self,
        periods: int = 30,
        freq: str = 'D',
        include_history: bool = False
    ) -> pd.DataFrame:
        """
        Make predictions.
        
        Args:
            periods: Number of future periods to forecast
            freq: Frequency ('D' for daily, 'W' for weekly, etc.)
            include_history: Include historical predictions
            
        Returns:
            DataFrame with predictions and confidence intervals
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        future = self.model.make_future_dataframe(
            periods=periods,
            freq=freq,
            include_history=include_history
        )
        
        forecast = self.model.predict(future)
        
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    
    def predict_values(self, periods: int = 30) -> np.ndarray:
        """Get just the predicted values as array."""
        forecast = self.predict(periods=periods, include_history=False)
        return forecast['yhat'].values
    
    def get_components(self) -> pd.DataFrame:
        """Get trend and seasonality components."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        future = self.model.make_future_dataframe(periods=0)
        forecast = self.model.predict(future)
        
        return forecast
    
    def add_regressor(self, name: str, prior_scale: float = 10.0) -> None:
        """Add external regressor."""
        if self.model is None:
            self.build()
        self.model.add_regressor(name, prior_scale=prior_scale)
    
    def add_seasonality(
        self,
        name: str,
        period: float,
        fourier_order: int,
        prior_scale: float = 10.0
    ) -> None:
        """Add custom seasonality component."""
        if self.model is None:
            self.build()
        self.model.add_seasonality(
            name=name,
            period=period,
            fourier_order=fourier_order,
            prior_scale=prior_scale
        )
    
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
            return f"Prophet (seasonality_mode={self.seasonality_mode})"
        return "Prophet (not fitted)"


def create_prophet_model(**kwargs) -> ProphetModel:
    """Factory function to create Prophet model."""
    return ProphetModel(**kwargs)
