"""
Data Loader Module for Agricultural Commodity Price Prediction

This module handles all data loading, preprocessing, and sequence generation
for the deep learning price prediction models.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class DataLoader:
    """
    Comprehensive data loader for agricultural commodity price data.
    
    Handles:
    - Loading raw CSV data
    - Missing value imputation
    - Train/val/test splitting with temporal ordering
    - Sequence generation for time series models
    - Feature scaling and normalization
    """
    
    def __init__(
        self,
        data_path: str = None,
        sequence_length: int = 30,
        forecast_horizon: int = 1,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        scaler_type: str = 'minmax'
    ):
        """
        Initialize DataLoader with configuration parameters.
        
        Args:
            data_path: Path to the raw CSV file
            sequence_length: Number of time steps to look back
            forecast_horizon: Number of steps ahead to predict
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation
            test_ratio: Proportion of data for testing
            scaler_type: Type of scaler ('minmax' or 'standard')
        """
        self.data_path = data_path
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.scaler_type = scaler_type
        
        # Initialize scalers
        self.feature_scaler = None
        self.target_scaler = None
        
        # Data containers
        self.raw_data = None
        self.processed_data = None
        self.commodity_data = {}
        
    def load_data(self, filepath: str = None) -> pd.DataFrame:
        """
        Load raw data from CSV file.
        
        Args:
            filepath: Path to CSV file (overrides init path)
            
        Returns:
            DataFrame with loaded data
        """
        path = filepath or self.data_path
        if path is None:
            raise ValueError("No data path provided")
            
        self.raw_data = pd.read_csv(path)
        print(f"✓ Loaded {len(self.raw_data)} records from {path}")
        print(f"  Columns: {list(self.raw_data.columns)}")
        
        return self.raw_data
    
    def preprocess_data(
        self,
        df: pd.DataFrame = None,
        commodity: str = None,
        state: str = None,
        market: str = None
    ) -> pd.DataFrame:
        """
        Preprocess the raw data with cleaning and filtering.
        
        Args:
            df: DataFrame to preprocess (uses raw_data if None)
            commodity: Filter for specific commodity
            state: Filter for specific state
            market: Filter for specific market
            
        Returns:
            Preprocessed DataFrame
        """
        data = df.copy() if df is not None else self.raw_data.copy()
        
        # Standardize column names
        data.columns = data.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # Parse dates - try multiple formats
        date_cols = [col for col in data.columns if 'date' in col.lower() or 'arrival' in col.lower()]
        if date_cols:
            for col in date_cols:
                # Try DD-MM-YYYY format first (matches the dataset)
                try:
                    data[col] = pd.to_datetime(data[col], format='%d-%m-%Y', errors='coerce')
                except:
                    try:
                        data[col] = pd.to_datetime(data[col], format='%d/%m/%Y', errors='coerce')
                    except:
                        data[col] = pd.to_datetime(data[col], errors='coerce')
        
        # Identify the date column
        date_col = None
        for col in ['arrival_date', 'date', 'price_date']:
            if col in data.columns:
                date_col = col
                break
        
        if date_col:
            data['date'] = data[date_col]
        
        # Apply commodity filter BEFORE dropping NaT dates
        if commodity:
            commodity_lower = commodity.lower().strip()
            data = data[data['commodity'].str.lower().str.strip() == commodity_lower]
        if state:
            data = data[data['state'].str.lower().str.strip() == state.lower().strip()]
        if market:
            data = data[data['market'].str.lower().str.strip() == market.lower().strip()]
        
        # Drop NaT dates AFTER filtering
        if date_col and 'date' in data.columns:
            data = data.dropna(subset=['date'])
            data = data.sort_values('date')
        
        # Handle price columns
        price_cols = ['min_price', 'max_price', 'modal_price', 
                     'min_x0020_price', 'max_x0020_price', 'modal_x0020_price']
        
        for col in price_cols:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Rename price columns if they exist with x0020
        rename_map = {
            'min_x0020_price': 'min_price',
            'max_x0020_price': 'max_price',
            'modal_x0020_price': 'modal_price'
        }
        data = data.rename(columns={k: v for k, v in rename_map.items() if k in data.columns})
        
        # Handle missing values
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        data[numeric_cols] = data[numeric_cols].fillna(method='ffill').fillna(method='bfill')
        
        # Remove outliers using IQR method
        for col in ['min_price', 'max_price', 'modal_price']:
            if col in data.columns:
                Q1 = data[col].quantile(0.01)
                Q3 = data[col].quantile(0.99)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                data = data[(data[col] >= lower) & (data[col] <= upper)]
        
        self.processed_data = data
        print(f"✓ Preprocessed data: {len(data)} records")
        
        return data
    
    def get_commodity_list(self) -> List[str]:
        """Get list of unique commodities in the dataset."""
        if self.raw_data is None:
            return []
        
        col = 'commodity' if 'commodity' in self.raw_data.columns else 'Commodity'
        return self.raw_data[col].unique().tolist()
    
    def aggregate_daily_prices(
        self,
        df: pd.DataFrame,
        price_col: str = 'modal_price',
        agg_method: str = 'mean'
    ) -> pd.DataFrame:
        """
        Aggregate prices to daily level.
        
        Args:
            df: Input DataFrame
            price_col: Column to aggregate
            agg_method: Aggregation method ('mean', 'median', 'weighted')
            
        Returns:
            Daily aggregated DataFrame
        """
        if 'date' not in df.columns:
            raise ValueError("DataFrame must have 'date' column")
        
        # Group by date and aggregate
        if agg_method == 'mean':
            daily = df.groupby('date')[price_col].mean().reset_index()
        elif agg_method == 'median':
            daily = df.groupby('date')[price_col].median().reset_index()
        elif agg_method == 'weighted':
            # Weight by volume if available
            if 'arrivals' in df.columns:
                daily = df.groupby('date').apply(
                    lambda x: np.average(x[price_col], weights=x['arrivals'] + 1)
                ).reset_index(name=price_col)
            else:
                daily = df.groupby('date')[price_col].mean().reset_index()
        else:
            raise ValueError(f"Unknown aggregation method: {agg_method}")
        
        daily = daily.set_index('date')
        daily = daily.sort_index()
        
        # Fill missing dates
        date_range = pd.date_range(start=daily.index.min(), end=daily.index.max(), freq='D')
        daily = daily.reindex(date_range)
        daily[price_col] = daily[price_col].interpolate(method='linear')
        daily = daily.reset_index().rename(columns={'index': 'date'})
        
        return daily
    
    def create_sequences(
        self,
        data: np.ndarray,
        target_col_idx: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series prediction.
        
        Args:
            data: Input data array (samples, features)
            target_col_idx: Index of target column
            
        Returns:
            Tuple of (X sequences, y targets)
        """
        X, y = [], []
        
        for i in range(len(data) - self.sequence_length - self.forecast_horizon + 1):
            X.append(data[i:(i + self.sequence_length)])
            y.append(data[i + self.sequence_length + self.forecast_horizon - 1, target_col_idx])
        
        return np.array(X), np.array(y)
    
    def prepare_data_for_training(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = 'modal_price'
    ) -> Dict[str, np.ndarray]:
        """
        Prepare complete dataset for model training.
        
        Args:
            df: Input DataFrame with features
            feature_cols: List of feature column names
            target_col: Target column name
            
        Returns:
            Dictionary with train/val/test X and y arrays
        """
        # Ensure target is in feature columns for sequence creation
        all_cols = [target_col] + [c for c in feature_cols if c != target_col]
        
        # Extract data
        data = df[all_cols].values
        
        # Initialize scalers
        if self.scaler_type == 'minmax':
            self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
            self.target_scaler = MinMaxScaler(feature_range=(0, 1))
        else:
            self.feature_scaler = StandardScaler()
            self.target_scaler = StandardScaler()
        
        # Scale features
        scaled_data = self.feature_scaler.fit_transform(data)
        
        # Fit target scaler separately for inverse transform
        self.target_scaler.fit(df[[target_col]].values)
        
        # Create sequences
        X, y = self.create_sequences(scaled_data, target_col_idx=0)
        
        # Calculate split indices
        n_samples = len(X)
        train_end = int(n_samples * self.train_ratio)
        val_end = int(n_samples * (self.train_ratio + self.val_ratio))
        
        # Split data (temporal order preserved)
        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]
        
        print(f"✓ Data prepared for training:")
        print(f"  Training:   {X_train.shape[0]} samples")
        print(f"  Validation: {X_val.shape[0]} samples")
        print(f"  Test:       {X_test.shape[0]} samples")
        print(f"  Sequence length: {self.sequence_length}")
        print(f"  Features: {len(all_cols)}")
        
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test,
            'feature_cols': all_cols,
            'dates_test': df.iloc[val_end + self.sequence_length:]['date'].values if 'date' in df.columns else None
        }
    
    def inverse_transform_predictions(
        self,
        predictions: np.ndarray
    ) -> np.ndarray:
        """
        Inverse transform scaled predictions to original scale.
        
        Args:
            predictions: Scaled prediction values
            
        Returns:
            Predictions in original scale
        """
        if self.target_scaler is None:
            raise ValueError("Target scaler not fitted. Run prepare_data_for_training first.")
        
        predictions = predictions.reshape(-1, 1)
        return self.target_scaler.inverse_transform(predictions).flatten()
    
    def create_time_series_cv_splits(
        self,
        n_splits: int = 5,
        gap: int = 0
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create time series cross-validation splits.
        
        Args:
            n_splits: Number of CV splits
            gap: Gap between train and test to prevent leakage
            
        Returns:
            List of (train_indices, test_indices) tuples
        """
        if self.processed_data is None:
            raise ValueError("No processed data available")
        
        n_samples = len(self.processed_data)
        fold_size = n_samples // (n_splits + 1)
        
        splits = []
        for i in range(n_splits):
            train_end = fold_size * (i + 1)
            test_start = train_end + gap
            test_end = test_start + fold_size
            
            train_idx = np.arange(0, train_end)
            test_idx = np.arange(test_start, min(test_end, n_samples))
            
            splits.append((train_idx, test_idx))
        
        return splits


class SlidingWindowGenerator:
    """
    Generator for efficient batch loading of time series sequences.
    Uses sliding window approach for memory efficiency.
    """
    
    def __init__(
        self,
        data: np.ndarray,
        sequence_length: int,
        batch_size: int,
        target_col_idx: int = 0,
        shuffle: bool = False
    ):
        """
        Initialize generator.
        
        Args:
            data: Input data array
            sequence_length: Length of input sequences
            batch_size: Batch size for training
            target_col_idx: Index of target column
            shuffle: Whether to shuffle batches
        """
        self.data = data
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.target_col_idx = target_col_idx
        self.shuffle = shuffle
        
        self.n_samples = len(data) - sequence_length
        self.indices = np.arange(self.n_samples)
        
    def __len__(self):
        return int(np.ceil(self.n_samples / self.batch_size))
    
    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        
        for start_idx in range(0, self.n_samples, self.batch_size):
            batch_indices = self.indices[start_idx:start_idx + self.batch_size]
            
            X_batch = np.array([
                self.data[i:i + self.sequence_length]
                for i in batch_indices
            ])
            
            y_batch = np.array([
                self.data[i + self.sequence_length, self.target_col_idx]
                for i in batch_indices
            ])
            
            yield X_batch, y_batch
    
    def get_dataset(self):
        """Get complete dataset as numpy arrays."""
        X, y = [], []
        for X_batch, y_batch in self:
            X.append(X_batch)
            y.append(y_batch)
        return np.vstack(X), np.concatenate(y)
