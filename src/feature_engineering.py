"""
Feature Engineering Module for Agricultural Commodity Price Prediction

This module implements comprehensive feature engineering including:
- Temporal features (day, week, month, quarter, year, seasonal indicators)
- Lag features for capturing historical patterns
- Rolling statistics (moving averages, volatility)
- Technical indicators (RSI, MACD, Bollinger Bands)
- External proxy features (weather seasonality, inflation)
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    Comprehensive feature engineering for time series price prediction.
    
    Creates features in several categories:
    1. Temporal: Calendar-based features
    2. Lag: Historical price values
    3. Rolling: Moving averages and statistics
    4. Technical: Trading indicators
    5. External: Weather and economic proxies
    """
    
    def __init__(
        self,
        target_col: str = 'modal_price',
        lag_periods: List[int] = None,
        rolling_windows: List[int] = None
    ):
        """
        Initialize FeatureEngineer.
        
        Args:
            target_col: Name of target price column
            lag_periods: List of lag periods to create
            rolling_windows: List of rolling window sizes
        """
        self.target_col = target_col
        self.lag_periods = lag_periods or [1, 2, 3, 5, 7, 14, 21, 30]
        self.rolling_windows = rolling_windows or [3, 7, 14, 21, 30]
        self.feature_names = []
        
    def create_all_features(
        self,
        df: pd.DataFrame,
        include_external: bool = True
    ) -> pd.DataFrame:
        """
        Create all engineered features.
        
        Args:
            df: Input DataFrame with date and price columns
            include_external: Whether to include external proxy features
            
        Returns:
            DataFrame with all features added
        """
        data = df.copy()
        
        # Ensure date column is datetime
        if 'date' in data.columns and not pd.api.types.is_datetime64_any_dtype(data['date']):
            data['date'] = pd.to_datetime(data['date'])
        
        # Create features in order
        print("Creating temporal features...")
        data = self.create_temporal_features(data)
        
        print("Creating lag features...")
        data = self.create_lag_features(data)
        
        print("Creating rolling statistics...")
        data = self.create_rolling_features(data)
        
        print("Creating technical indicators...")
        data = self.create_technical_indicators(data)
        
        if include_external:
            print("Creating external proxy features...")
            data = self.create_external_features(data)
        
        # Handle any remaining missing values
        data = self._handle_missing_values(data)
        
        print(f"✓ Created {len(self.feature_names)} features")
        
        return data
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create calendar and time-based features.
        
        Features created:
        - Day of week, month, quarter, year
        - Week of year
        - Is weekend, is month start/end
        - Season indicators
        - Indian festival season proxies
        """
        data = df.copy()
        
        if 'date' not in data.columns:
            return data
        
        date = data['date']
        
        # Basic temporal features
        data['day_of_week'] = date.dt.dayofweek
        data['day_of_month'] = date.dt.day
        data['day_of_year'] = date.dt.dayofyear
        data['week_of_year'] = date.dt.isocalendar().week.astype(int)
        data['month'] = date.dt.month
        data['quarter'] = date.dt.quarter
        data['year'] = date.dt.year
        
        # Binary features
        data['is_weekend'] = (date.dt.dayofweek >= 5).astype(int)
        data['is_month_start'] = date.dt.is_month_start.astype(int)
        data['is_month_end'] = date.dt.is_month_end.astype(int)
        data['is_quarter_start'] = date.dt.is_quarter_start.astype(int)
        data['is_quarter_end'] = date.dt.is_quarter_end.astype(int)
        
        # Season encoding (Indian agricultural seasons)
        # Kharif (Jun-Oct), Rabi (Nov-Mar), Zaid (Apr-May)
        data['season_kharif'] = data['month'].isin([6, 7, 8, 9, 10]).astype(int)
        data['season_rabi'] = data['month'].isin([11, 12, 1, 2, 3]).astype(int)
        data['season_zaid'] = data['month'].isin([4, 5]).astype(int)
        
        # Indian festival season indicator (approx Oct-Nov: Diwali, etc.)
        data['is_festival_season'] = data['month'].isin([10, 11]).astype(int)
        
        # Cyclical encoding for month and day
        data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
        data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
        data['day_of_week_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
        data['day_of_week_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
        data['day_of_year_sin'] = np.sin(2 * np.pi * data['day_of_year'] / 365)
        data['day_of_year_cos'] = np.cos(2 * np.pi * data['day_of_year'] / 365)
        
        temporal_features = [
            'day_of_week', 'day_of_month', 'day_of_year', 'week_of_year',
            'month', 'quarter', 'year', 'is_weekend', 'is_month_start',
            'is_month_end', 'is_quarter_start', 'is_quarter_end',
            'season_kharif', 'season_rabi', 'season_zaid', 'is_festival_season',
            'month_sin', 'month_cos', 'day_of_week_sin', 'day_of_week_cos',
            'day_of_year_sin', 'day_of_year_cos'
        ]
        self.feature_names.extend(temporal_features)
        
        return data
    
    def create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create lagged price features to capture historical patterns.
        
        Features created:
        - Lagged prices at various periods
        - Price differences (momentum)
        - Percentage changes
        """
        data = df.copy()
        target = self.target_col
        
        for lag in self.lag_periods:
            # Lagged price
            data[f'price_lag_{lag}'] = data[target].shift(lag)
            self.feature_names.append(f'price_lag_{lag}')
            
            # Price difference
            data[f'price_diff_{lag}'] = data[target] - data[target].shift(lag)
            self.feature_names.append(f'price_diff_{lag}')
            
            # Percentage change
            data[f'price_pct_change_{lag}'] = data[target].pct_change(periods=lag) * 100
            self.feature_names.append(f'price_pct_change_{lag}')
        
        # Momentum features
        data['momentum_short'] = data[target] - data[target].shift(3)
        data['momentum_medium'] = data[target] - data[target].shift(7)
        data['momentum_long'] = data[target] - data[target].shift(14)
        
        self.feature_names.extend(['momentum_short', 'momentum_medium', 'momentum_long'])
        
        return data
    
    def create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create rolling window statistics.
        
        Features created:
        - Moving averages
        - Rolling standard deviation (volatility)
        - Rolling min/max
        - Rolling median
        - Exponential moving averages
        """
        data = df.copy()
        target = self.target_col
        
        for window in self.rolling_windows:
            # Simple moving average
            data[f'sma_{window}'] = data[target].rolling(window=window, min_periods=1).mean()
            
            # Rolling standard deviation
            data[f'rolling_std_{window}'] = data[target].rolling(window=window, min_periods=1).std()
            
            # Rolling min/max
            data[f'rolling_min_{window}'] = data[target].rolling(window=window, min_periods=1).min()
            data[f'rolling_max_{window}'] = data[target].rolling(window=window, min_periods=1).max()
            
            # Rolling median
            data[f'rolling_median_{window}'] = data[target].rolling(window=window, min_periods=1).median()
            
            # Price range
            data[f'price_range_{window}'] = data[f'rolling_max_{window}'] - data[f'rolling_min_{window}']
            
            # Price relative to moving average
            data[f'price_to_sma_{window}'] = data[target] / (data[f'sma_{window}'] + 1e-8)
            
            # Rolling skewness and kurtosis (min_periods must be <= window)
            min_p_skew = min(3, window)
            min_p_kurt = min(4, window)
            data[f'rolling_skew_{window}'] = data[target].rolling(window=window, min_periods=min_p_skew).skew()
            data[f'rolling_kurt_{window}'] = data[target].rolling(window=window, min_periods=min_p_kurt).kurt()
            
            # Track features
            self.feature_names.extend([
                f'sma_{window}', f'rolling_std_{window}', f'rolling_min_{window}',
                f'rolling_max_{window}', f'rolling_median_{window}', f'price_range_{window}',
                f'price_to_sma_{window}', f'rolling_skew_{window}', f'rolling_kurt_{window}'
            ])
        
        # Exponential moving averages
        for span in [7, 14, 21]:
            data[f'ema_{span}'] = data[target].ewm(span=span, adjust=False).mean()
            self.feature_names.append(f'ema_{span}')
        
        # EMA crossovers
        data['ema_7_21_ratio'] = data['ema_7'] / (data['ema_21'] + 1e-8)
        data['ema_crossover'] = (data['ema_7'] > data['ema_21']).astype(int)
        
        self.feature_names.extend(['ema_7_21_ratio', 'ema_crossover'])
        
        return data
    
    def create_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create technical trading indicators.
        
        Features created:
        - RSI (Relative Strength Index)
        - MACD (Moving Average Convergence Divergence)
        - Bollinger Bands
        - Rate of Change
        - Average True Range proxy
        """
        data = df.copy()
        target = self.target_col
        
        # RSI (Relative Strength Index)
        data = self._calculate_rsi(data, target, periods=[7, 14, 21])
        
        # MACD
        data = self._calculate_macd(data, target)
        
        # Bollinger Bands
        data = self._calculate_bollinger_bands(data, target, window=20, num_std=2)
        
        # Rate of Change
        for period in [7, 14, 30]:
            data[f'roc_{period}'] = ((data[target] - data[target].shift(period)) / 
                                     (data[target].shift(period) + 1e-8)) * 100
            self.feature_names.append(f'roc_{period}')
        
        # Commodity Channel Index (CCI) proxy
        data = self._calculate_cci(data, target, window=20)
        
        # Stochastic oscillator proxy
        data = self._calculate_stochastic(data, target, window=14)
        
        return data
    
    def _calculate_rsi(
        self,
        df: pd.DataFrame,
        target: str,
        periods: List[int] = [14]
    ) -> pd.DataFrame:
        """Calculate RSI for given periods."""
        data = df.copy()
        
        for period in periods:
            delta = data[target].diff()
            gain = delta.where(delta > 0, 0)
            loss = (-delta).where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=period, min_periods=1).mean()
            avg_loss = loss.rolling(window=period, min_periods=1).mean()
            
            rs = avg_gain / (avg_loss + 1e-8)
            data[f'rsi_{period}'] = 100 - (100 / (1 + rs))
            
            self.feature_names.append(f'rsi_{period}')
        
        return data
    
    def _calculate_macd(
        self,
        df: pd.DataFrame,
        target: str,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> pd.DataFrame:
        """Calculate MACD indicator."""
        data = df.copy()
        
        ema_fast = data[target].ewm(span=fast, adjust=False).mean()
        ema_slow = data[target].ewm(span=slow, adjust=False).mean()
        
        data['macd'] = ema_fast - ema_slow
        data['macd_signal'] = data['macd'].ewm(span=signal, adjust=False).mean()
        data['macd_histogram'] = data['macd'] - data['macd_signal']
        
        # MACD crossover signals
        data['macd_crossover'] = (data['macd'] > data['macd_signal']).astype(int)
        
        self.feature_names.extend(['macd', 'macd_signal', 'macd_histogram', 'macd_crossover'])
        
        return data
    
    def _calculate_bollinger_bands(
        self,
        df: pd.DataFrame,
        target: str,
        window: int = 20,
        num_std: int = 2
    ) -> pd.DataFrame:
        """Calculate Bollinger Bands."""
        data = df.copy()
        
        rolling_mean = data[target].rolling(window=window, min_periods=1).mean()
        rolling_std = data[target].rolling(window=window, min_periods=1).std()
        
        data['bb_middle'] = rolling_mean
        data['bb_upper'] = rolling_mean + (rolling_std * num_std)
        data['bb_lower'] = rolling_mean - (rolling_std * num_std)
        
        # Bollinger Band width
        data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / (data['bb_middle'] + 1e-8)
        
        # %B indicator (where price is relative to bands)
        data['bb_pct_b'] = (data[target] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'] + 1e-8)
        
        self.feature_names.extend(['bb_middle', 'bb_upper', 'bb_lower', 'bb_width', 'bb_pct_b'])
        
        return data
    
    def _calculate_cci(
        self,
        df: pd.DataFrame,
        target: str,
        window: int = 20
    ) -> pd.DataFrame:
        """Calculate Commodity Channel Index."""
        data = df.copy()
        
        # For single price, use target as typical price
        typical_price = data[target]
        sma = typical_price.rolling(window=window, min_periods=1).mean()
        mean_dev = typical_price.rolling(window=window, min_periods=1).apply(
            lambda x: np.abs(x - x.mean()).mean(), raw=True
        )
        
        data['cci'] = (typical_price - sma) / (0.015 * mean_dev + 1e-8)
        
        self.feature_names.append('cci')
        
        return data
    
    def _calculate_stochastic(
        self,
        df: pd.DataFrame,
        target: str,
        window: int = 14
    ) -> pd.DataFrame:
        """Calculate Stochastic Oscillator."""
        data = df.copy()
        
        low_min = data[target].rolling(window=window, min_periods=1).min()
        high_max = data[target].rolling(window=window, min_periods=1).max()
        
        data['stoch_k'] = 100 * (data[target] - low_min) / (high_max - low_min + 1e-8)
        data['stoch_d'] = data['stoch_k'].rolling(window=3, min_periods=1).mean()
        
        self.feature_names.extend(['stoch_k', 'stoch_d'])
        
        return data
    
    def create_external_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create external proxy features.
        
        Since we don't have actual weather/economic data,
        we create meaningful proxies based on patterns.
        
        Features created:
        - Seasonal weather index
        - Inflation proxy
        - Market demand index
        - Supply disruption proxy
        """
        data = df.copy()
        target = self.target_col
        
        # Seasonal Weather Index (simulated based on month)
        # Higher during monsoon (Jun-Sep), affects agricultural prices
        if 'month' in data.columns:
            weather_map = {
                1: 0.3, 2: 0.35, 3: 0.5, 4: 0.7, 5: 0.85,
                6: 0.95, 7: 1.0, 8: 0.95, 9: 0.8, 10: 0.5,
                11: 0.35, 12: 0.3
            }
            data['weather_index'] = data['month'].map(weather_map)
            
            # Temperature proxy (higher in summer)
            temp_map = {
                1: 0.3, 2: 0.4, 3: 0.6, 4: 0.8, 5: 0.95,
                6: 0.9, 7: 0.75, 8: 0.7, 9: 0.65, 10: 0.5,
                11: 0.35, 12: 0.3
            }
            data['temp_index'] = data['month'].map(temp_map)
        
        # Inflation Proxy (rolling price level changes)
        data['inflation_proxy_30d'] = data[target].pct_change(periods=30).rolling(window=30, min_periods=1).mean() * 100
        data['inflation_proxy_90d'] = data[target].pct_change(periods=90).rolling(window=30, min_periods=1).mean() * 100
        
        # Market Demand Index (based on price deviations from trend)
        trend = data[target].rolling(window=30, min_periods=1).mean()
        data['demand_index'] = (data[target] - trend) / (trend + 1e-8)
        
        # Volatility as supply disruption proxy
        data['supply_disruption_proxy'] = data[target].rolling(window=7, min_periods=1).std() / \
                                          data[target].rolling(window=30, min_periods=1).std().mean()
        
        # Global market proxy (lagged correlation with self)
        data['global_market_proxy'] = data[target].rolling(window=14, min_periods=1).corr(
            data[target].shift(7)
        )
        
        # Price pressure indicator
        data['price_pressure'] = (data['ema_7'] - data['ema_21']) / (data['ema_21'] + 1e-8) \
                                 if 'ema_7' in data.columns else 0
        
        self.feature_names.extend([
            'weather_index', 'temp_index', 'inflation_proxy_30d', 'inflation_proxy_90d',
            'demand_index', 'supply_disruption_proxy', 'global_market_proxy', 'price_pressure'
        ])
        
        return data
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in engineered features."""
        data = df.copy()
        
        # Forward fill then backward fill
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        data[numeric_cols] = data[numeric_cols].fillna(method='ffill')
        data[numeric_cols] = data[numeric_cols].fillna(method='bfill')
        
        # Replace any remaining NaN/Inf with 0
        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.fillna(0)
        
        return data
    
    def get_feature_names(self, include_target: bool = False) -> List[str]:
        """
        Get list of all feature names.
        
        Args:
            include_target: Whether to include target column
            
        Returns:
            List of feature names
        """
        features = list(set(self.feature_names))
        if include_target:
            features = [self.target_col] + features
        return features
    
    def select_top_features(
        self,
        df: pd.DataFrame,
        n_features: int = 50,
        method: str = 'correlation'
    ) -> List[str]:
        """
        Select top features based on correlation with target.
        
        Args:
            df: DataFrame with features
            n_features: Number of features to select
            method: Selection method ('correlation', 'mutual_info')
            
        Returns:
            List of top feature names
        """
        feature_cols = [c for c in self.feature_names if c in df.columns]
        
        if method == 'correlation':
            correlations = {}
            for col in feature_cols:
                if col != self.target_col:
                    corr = abs(df[col].corr(df[self.target_col]))
                    if not np.isnan(corr):
                        correlations[col] = corr
            
            sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
            top_features = [f[0] for f in sorted_features[:n_features]]
            
            print(f"✓ Selected top {len(top_features)} features by correlation")
            
            return top_features
        
        return feature_cols[:n_features]


def create_feature_importance_plot(
    feature_names: List[str],
    importances: np.ndarray,
    top_n: int = 30,
    figsize: Tuple[int, int] = (12, 10)
):
    """
    Create feature importance visualization.
    
    Args:
        feature_names: List of feature names
        importances: Array of importance scores
        top_n: Number of top features to plot
        figsize: Figure size
    """
    import matplotlib.pyplot as plt
    
    # Sort by importance
    indices = np.argsort(importances)[::-1][:top_n]
    
    plt.figure(figsize=figsize)
    plt.barh(range(top_n), importances[indices], align='center')
    plt.yticks(range(top_n), [feature_names[i] for i in indices])
    plt.xlabel('Feature Importance')
    plt.title(f'Top {top_n} Feature Importances')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    return plt.gcf()
