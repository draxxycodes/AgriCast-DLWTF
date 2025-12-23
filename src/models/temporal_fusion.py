"""
Temporal Fusion Transformer (TFT) Model

State-of-the-art architecture for interpretable multi-horizon forecasting.

Paper: "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting"
       Lim et al., 2021
"""

import tensorflow as tf
import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import *
import numpy as np
from typing import Dict, Any, Tuple

from .base_model import DeepLearningModel


@keras.saving.register_keras_serializable(package='tft')
class GatedLinearUnit(layers.Layer):
    """Gated Linear Unit for variable selection."""
    
    def __init__(self, units: int, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.dropout_rate = dropout
        
    def build(self, input_shape):
        self.linear = Dense(self.units)
        self.gate = Dense(self.units, activation='sigmoid')
        self.dropout = Dropout(self.dropout_rate)
        self.norm = LayerNormalization()
        
    def call(self, x, training=None):
        linear_out = self.linear(x)
        gate_out = self.gate(x)
        gated = linear_out * gate_out
        gated = self.dropout(gated, training=training)
        return self.norm(gated)
    
    def get_config(self):
        config = super().get_config()
        config.update({'units': self.units, 'dropout': self.dropout_rate})
        return config


class TFTModel(DeepLearningModel):
    """
    Temporal Fusion Transformer.
    
    Features:
    - Variable selection networks
    - Static covariate encoders
    - Temporal processing with LSTM
    - Multi-head attention for temporal patterns
    - Gated residual connections
    
    Parameters: ~2-3M depending on configuration
    """
    
    def __init__(
        self,
        sequence_length: int,
        n_features: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_lstm_layers: int = 2,
        dropout: float = 0.1,
        learning_rate: float = 0.0001,
        **kwargs
    ):
        super().__init__("Temporal Fusion Transformer", sequence_length, n_features, **kwargs)
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_lstm_layers = num_lstm_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        
    def build(self) -> None:
        """Build the TFT architecture."""
        inputs = Input(shape=(self.sequence_length, self.n_features), name='input')
        
        # Input embedding
        x = Dense(self.d_model)(inputs)
        x = LayerNormalization()(x)
        
        # Bidirectional LSTM encoder
        for i in range(self.num_lstm_layers):
            x = Bidirectional(LSTM(self.d_model, return_sequences=True, dropout=self.dropout))(x)
            x = LayerNormalization()(x)
        
        # Static enrichment
        static = GlobalAveragePooling1D()(x)
        static = Dense(self.d_model, activation='relu')(static)
        static = RepeatVector(self.sequence_length)(static)
        x = Concatenate()([x, static])
        x = Dense(self.d_model)(x)
        
        # Temporal self-attention
        attn = layers.MultiHeadAttention(
            num_heads=self.num_heads, 
            key_dim=self.d_model // self.num_heads
        )(x, x)
        x = LayerNormalization()(x + attn)
        
        # Gated skip connection
        x = GatedLinearUnit(self.d_model, self.dropout)(x)
        
        # Feed-forward
        ff = Dense(self.d_model * 4, activation='gelu')(x)
        ff = Dropout(self.dropout)(ff)
        ff = Dense(self.d_model)(ff)
        x = LayerNormalization()(x + ff)
        
        # Output
        x = GlobalAveragePooling1D()(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(self.dropout)(x)
        outputs = Dense(1, dtype='float32')(x)
        
        self.model = Model(inputs, outputs, name='TFT')
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
            return f"TFT: {self.model.count_params():,} parameters"
        return "TFT (not built)"


def create_tft_model(sequence_length: int, n_features: int, **kwargs) -> TFTModel:
    """Factory function to create TFT model."""
    model = TFTModel(sequence_length, n_features, **kwargs)
    model.build()
    return model
