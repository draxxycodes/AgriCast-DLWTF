"""
N-BEATS Model (Neural Basis Expansion Analysis for Time Series)

Deep interpretable architecture for time series forecasting.

Paper: "N-BEATS: Neural basis expansion analysis for interpretable time series forecasting"
       Oreshkin et al., 2020
"""

import tensorflow as tf
import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import *
import numpy as np
from typing import Dict, Any, List

from .base_model import DeepLearningModel


class NBEATSModel(DeepLearningModel):
    """
    N-BEATS - Neural Basis Expansion Analysis.
    
    Architecture:
    - Stack of blocks with fully connected layers
    - Each block outputs backcast and forecast
    - Residual connections between blocks
    - Interpretable decomposition of forecast
    
    Parameters: ~3-5M depending on configuration
    """
    
    def __init__(
        self,
        sequence_length: int,
        n_features: int,
        num_stacks: int = 2,
        num_blocks: int = 3,
        hidden_units: int = 256,
        theta_dim: int = 128,
        num_layers_per_block: int = 4,
        dropout: float = 0.1,
        learning_rate: float = 0.0001,
        **kwargs
    ):
        super().__init__("N-BEATS", sequence_length, n_features, **kwargs)
        
        self.num_stacks = num_stacks
        self.num_blocks = num_blocks
        self.hidden_units = hidden_units
        self.theta_dim = theta_dim
        self.num_layers_per_block = num_layers_per_block
        self.dropout = dropout
        self.learning_rate = learning_rate
        
    def build(self) -> None:
        """Build the N-BEATS architecture."""
        inputs = Input(shape=(self.sequence_length, self.n_features), name='input')
        
        # Flatten input
        x = Flatten()(inputs)
        input_dim = self.sequence_length * self.n_features
        
        residual = x
        forecasts = []
        
        for stack_idx in range(self.num_stacks):
            for block_idx in range(self.num_blocks):
                # Fully connected block
                h = residual
                
                for layer_idx in range(self.num_layers_per_block):
                    h = Dense(self.hidden_units, activation='relu')(h)
                    h = BatchNormalization()(h)
                    h = Dropout(self.dropout)(h)
                
                # Theta for backcast and forecast
                theta_b = Dense(self.theta_dim)(h)
                theta_f = Dense(self.theta_dim)(h)
                
                # Backcast (reconstruct input)
                backcast = Dense(input_dim)(theta_b)
                
                # Forecast
                forecast = Dense(32, activation='relu')(theta_f)
                forecast = Dense(1)(forecast)
                
                # Update residual
                residual = residual - backcast
                forecasts.append(forecast)
        
        # Sum all forecasts
        if len(forecasts) > 1:
            output = Add()(forecasts)
        else:
            output = forecasts[0]
        
        outputs = Dense(1, dtype='float32')(output)
        
        self.model = Model(inputs, outputs, name='NBEATS')
        self.model.compile(
            optimizer=keras.optimizers.AdamW(
                learning_rate=self.learning_rate,
                weight_decay=1e-5
            ),
            loss='huber',
            metrics=['mae', 'mse']
        )
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        epochs: int = 100,
        batch_size: int = 32,
        callbacks: list = None,
        verbose: int = 1
    ) -> Dict[str, Any]:
        """Train the model."""
        if self.model is None:
            self.build()
        
        validation_data = (X_val, y_val) if X_val is not None else None
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        self.is_fitted = True
        return history.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.predict(X, verbose=0).flatten()
    
    def save(self, path: str) -> None:
        """Save model."""
        self.model.save(path)
    
    def load(self, path: str) -> None:
        """Load model."""
        self.model = keras.models.load_model(path)
        self.is_fitted = True
    
    def summary(self) -> str:
        """Get model summary."""
        if self.model:
            return f"N-BEATS: {self.model.count_params():,} parameters"
        return "N-BEATS (not built)"


def create_nbeats_model(sequence_length: int, n_features: int, **kwargs) -> NBEATSModel:
    """Factory function to create N-BEATS model."""
    model = NBEATSModel(sequence_length, n_features, **kwargs)
    model.build()
    return model
