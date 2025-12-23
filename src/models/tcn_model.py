"""
TCN Model (Temporal Convolutional Network)

Residual convolutional architecture for sequence modeling.

Paper: "An Empirical Evaluation of Generic Convolutional and Recurrent Networks 
        for Sequence Modeling"
       Bai et al., 2018
"""

import tensorflow as tf
import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import *
import numpy as np
from typing import Dict, Any, List

from .base_model import DeepLearningModel


class TCNModel(DeepLearningModel):
    """
    Temporal Convolutional Network.
    
    Architecture:
    - Stacked residual blocks with dilated convolutions
    - Causal padding for sequence modeling
    - Batch normalization and dropout
    - Global pooling for output
    
    Parameters: ~2-3M depending on configuration
    """
    
    def __init__(
        self,
        sequence_length: int,
        n_features: int,
        num_channels: List[int] = None,
        kernel_size: int = 3,
        dropout: float = 0.1,
        learning_rate: float = 0.0001,
        **kwargs
    ):
        super().__init__("TCN", sequence_length, n_features, **kwargs)
        
        self.num_channels = num_channels or [64, 128, 128, 256]
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.learning_rate = learning_rate
        
    def _residual_block(self, x, channels: int, dilation: int):
        """Create a residual block with two convolutions."""
        # First convolution
        conv1 = Conv1D(
            channels, 
            self.kernel_size, 
            padding='causal',
            dilation_rate=dilation, 
            activation='relu'
        )(x)
        conv1 = BatchNormalization()(conv1)
        conv1 = Dropout(self.dropout)(conv1)
        
        # Second convolution
        conv2 = Conv1D(
            channels, 
            self.kernel_size, 
            padding='causal',
            dilation_rate=dilation, 
            activation='relu'
        )(conv1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Dropout(self.dropout)(conv2)
        
        # Residual connection
        if x.shape[-1] != channels:
            x = Conv1D(channels, kernel_size=1)(x)
        
        return Add()([x, conv2])
        
    def build(self) -> None:
        """Build the TCN architecture."""
        inputs = Input(shape=(self.sequence_length, self.n_features), name='input')
        
        x = inputs
        
        for i, channels in enumerate(self.num_channels):
            dilation = 2 ** i
            x = self._residual_block(x, channels, dilation)
            x = LayerNormalization()(x)
        
        # Dual pooling
        avg_pool = GlobalAveragePooling1D()(x)
        max_pool = GlobalMaxPooling1D()(x)
        x = Concatenate()([avg_pool, max_pool])
        
        # Output layers
        x = Dense(128, activation='relu')(x)
        x = Dropout(self.dropout)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(self.dropout)(x)
        outputs = Dense(1, dtype='float32')(x)
        
        self.model = Model(inputs, outputs, name='TCN')
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
            return f"TCN: {self.model.count_params():,} parameters"
        return "TCN (not built)"


def create_tcn_model(sequence_length: int, n_features: int, **kwargs) -> TCNModel:
    """Factory function to create TCN model."""
    model = TCNModel(sequence_length, n_features, **kwargs)
    model.build()
    return model
