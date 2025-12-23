"""
WaveNet Model for Time Series Forecasting

Dilated causal convolutions for sequence modeling.

Paper: "WaveNet: A Generative Model for Raw Audio"
       Van den Oord et al., 2016 (adapted for time series)
"""

import tensorflow as tf
import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import *
import numpy as np
from typing import Dict, Any

from .base_model import DeepLearningModel


class WaveNetModel(DeepLearningModel):
    """
    WaveNet with Dilated Causal Convolutions.
    
    Architecture:
    - Stacked dilated causal convolution blocks
    - Gated activation units
    - Skip connections for gradient flow
    - Exponentially increasing dilation rates
    
    Parameters: ~2M depending on configuration
    """
    
    def __init__(
        self,
        sequence_length: int,
        n_features: int,
        residual_channels: int = 64,
        skip_channels: int = 128,
        num_dilation_layers: int = 8,
        num_stacks: int = 2,
        kernel_size: int = 2,
        dropout: float = 0.1,
        learning_rate: float = 0.0001,
        **kwargs
    ):
        super().__init__("WaveNet", sequence_length, n_features, **kwargs)
        
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.num_dilation_layers = num_dilation_layers
        self.num_stacks = num_stacks
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.learning_rate = learning_rate
        
    def build(self) -> None:
        """Build the WaveNet architecture."""
        inputs = Input(shape=(self.sequence_length, self.n_features), name='input')
        
        # Initial causal convolution
        x = Conv1D(self.residual_channels, kernel_size=1)(inputs)
        
        skip_connections = []
        
        for stack in range(self.num_stacks):
            for i in range(self.num_dilation_layers):
                dilation = 2 ** i
                
                # Gated activation unit
                tanh_out = Conv1D(
                    self.residual_channels, 
                    kernel_size=self.kernel_size,
                    padding='causal', 
                    dilation_rate=dilation,
                    activation='tanh'
                )(x)
                
                sigmoid_out = Conv1D(
                    self.residual_channels, 
                    kernel_size=self.kernel_size,
                    padding='causal', 
                    dilation_rate=dilation,
                    activation='sigmoid'
                )(x)
                
                gated = tanh_out * sigmoid_out
                
                # Skip connection
                skip = Conv1D(self.skip_channels, kernel_size=1)(gated)
                skip_connections.append(skip)
                
                # Residual connection
                residual = Conv1D(self.residual_channels, kernel_size=1)(gated)
                x = Add()([x, residual])
                x = LayerNormalization()(x)
                x = Dropout(self.dropout)(x)
        
        # Combine skip connections
        skip_sum = Add()(skip_connections)
        skip_sum = keras.activations.relu(skip_sum)
        skip_sum = Conv1D(self.skip_channels, kernel_size=1, activation='relu')(skip_sum)
        skip_sum = Conv1D(128, kernel_size=1)(skip_sum)
        
        # Output layers
        x = GlobalAveragePooling1D()(skip_sum)
        x = Dense(64, activation='relu')(x)
        x = Dropout(self.dropout)(x)
        outputs = Dense(1, dtype='float32')(x)
        
        self.model = Model(inputs, outputs, name='WaveNet')
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
            return f"WaveNet: {self.model.count_params():,} parameters"
        return "WaveNet (not built)"


def create_wavenet_model(sequence_length: int, n_features: int, **kwargs) -> WaveNetModel:
    """Factory function to create WaveNet model."""
    model = WaveNetModel(sequence_length, n_features, **kwargs)
    model.build()
    return model
