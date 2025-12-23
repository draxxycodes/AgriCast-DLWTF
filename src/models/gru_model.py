"""
Bidirectional GRU Model with Residual Connections

This module implements a sophisticated GRU architecture with:
- Bidirectional GRU layers for capturing both past and future context
- Residual connections for better gradient flow
- Layer normalization for training stability
- Multi-scale feature extraction with pooling
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.layers import (
    Input, GRU, Bidirectional, Dense, Dropout, 
    LayerNormalization, BatchNormalization,
    GlobalAveragePooling1D, GlobalMaxPooling1D,
    Concatenate, Add, Multiply, Conv1D
)
import numpy as np
from typing import Tuple, Optional


class ResidualGRUBlock(layers.Layer):
    """
    Residual block with Bidirectional GRU.
    
    Implements skip connections around GRU layers for
    improved gradient flow during training.
    """
    
    def __init__(
        self,
        units: int,
        dropout_rate: float = 0.2,
        l2_reg: float = 0.001,
        **kwargs
    ):
        super(ResidualGRUBlock, self).__init__(**kwargs)
        self.units = units
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        
    def build(self, input_shape):
        self.gru = Bidirectional(
            GRU(
                self.units,
                return_sequences=True,
                kernel_regularizer=regularizers.l2(self.l2_reg),
                recurrent_regularizer=regularizers.l2(self.l2_reg),
                dropout=self.dropout_rate,
                recurrent_dropout=self.dropout_rate / 2
            )
        )
        self.norm = LayerNormalization()
        self.dropout = Dropout(self.dropout_rate)
        
        # Projection for skip connection if dimensions don't match
        self.projection = Dense(self.units * 2, use_bias=False)
        
        super(ResidualGRUBlock, self).build(input_shape)
    
    def call(self, inputs, training=None):
        # GRU forward pass
        gru_output = self.gru(inputs, training=training)
        gru_output = self.dropout(gru_output, training=training)
        
        # Project input for skip connection
        if inputs.shape[-1] != self.units * 2:
            residual = self.projection(inputs)
        else:
            residual = inputs
        
        # Add residual and normalize
        output = self.norm(gru_output + residual)
        
        return output
    
    def get_config(self):
        config = super(ResidualGRUBlock, self).get_config()
        config.update({
            'units': self.units,
            'dropout_rate': self.dropout_rate,
            'l2_reg': self.l2_reg
        })
        return config


class TemporalConvBlock(layers.Layer):
    """
    Temporal convolution block for local pattern extraction.
    
    Complements GRU by capturing local temporal patterns
    at different scales.
    """
    
    def __init__(
        self,
        filters: int = 64,
        kernel_sizes: Tuple[int, ...] = (3, 5, 7),
        dropout_rate: float = 0.2,
        **kwargs
    ):
        super(TemporalConvBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_sizes = kernel_sizes
        self.dropout_rate = dropout_rate
        
    def build(self, input_shape):
        self.convs = [
            Conv1D(
                filters=self.filters,
                kernel_size=k,
                padding='same',
                activation='relu'
            ) for k in self.kernel_sizes
        ]
        self.dropout = Dropout(self.dropout_rate)
        self.norm = LayerNormalization()
        
        super(TemporalConvBlock, self).build(input_shape)
    
    def call(self, inputs, training=None):
        # Multi-scale convolutions
        conv_outputs = [conv(inputs) for conv in self.convs]
        
        # Concatenate multi-scale features
        x = Concatenate()(conv_outputs)
        x = self.dropout(x, training=training)
        x = self.norm(x)
        
        return x
    
    def get_config(self):
        config = super(TemporalConvBlock, self).get_config()
        config.update({
            'filters': self.filters,
            'kernel_sizes': self.kernel_sizes,
            'dropout_rate': self.dropout_rate
        })
        return config


class BidirectionalGRUModel:
    """
    Bidirectional GRU with Residual Connections for time series forecasting.
    
    Architecture:
    - Input Layer
    - Temporal Convolution Block (multi-scale)
    - Bidirectional GRU Layer 1 with Residual (128 units)
    - Layer Normalization
    - Bidirectional GRU Layer 2 with Residual (64 units)
    - Global Average Pooling + Global Max Pooling (concatenated)
    - Dense layers with dropout
    - Output (1 unit for regression)
    
    Features:
    - Bidirectional processing captures both past and future context
    - Residual connections for better gradient flow
    - Multi-scale temporal convolutions
    - Dual pooling for comprehensive feature aggregation
    """
    
    def __init__(
        self,
        sequence_length: int = 30,
        n_features: int = 50,
        gru_units: Tuple[int, ...] = (128, 64, 32),
        conv_filters: int = 32,
        dense_units: Tuple[int, ...] = (128, 64),
        dropout_rate: float = 0.3,
        l2_reg: float = 0.001,
        learning_rate: float = 0.001
    ):
        """
        Initialize Bidirectional GRU model.
        
        Args:
            sequence_length: Number of time steps in input
            n_features: Number of input features
            gru_units: Tuple of units for each GRU layer
            conv_filters: Filters for temporal convolution
            dense_units: Tuple of units for dense layers
            dropout_rate: Dropout probability
            l2_reg: L2 regularization factor
            learning_rate: Initial learning rate
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.gru_units = gru_units
        self.conv_filters = conv_filters
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.learning_rate = learning_rate
        
        self.model = None
        self.history = None
        
    def build_model(self) -> Model:
        """
        Build the Bidirectional GRU architecture.
        
        Returns:
            Compiled Keras Model
        """
        # Input layer
        inputs = Input(shape=(self.sequence_length, self.n_features), name='input')
        
        # Temporal convolution for local pattern extraction
        x = TemporalConvBlock(
            filters=self.conv_filters,
            kernel_sizes=(3, 5, 7),
            dropout_rate=self.dropout_rate,
            name='temporal_conv'
        )(inputs)
        
        # First Bidirectional GRU with residual
        x = ResidualGRUBlock(
            units=self.gru_units[0],
            dropout_rate=self.dropout_rate,
            l2_reg=self.l2_reg,
            name='residual_gru_1'
        )(x)
        
        # Second Bidirectional GRU with residual
        x = ResidualGRUBlock(
            units=self.gru_units[1],
            dropout_rate=self.dropout_rate,
            l2_reg=self.l2_reg,
            name='residual_gru_2'
        )(x)
        
        # Third GRU layer (no residual, standard)
        x = Bidirectional(
            GRU(
                self.gru_units[2],
                return_sequences=True,
                kernel_regularizer=regularizers.l2(self.l2_reg),
                recurrent_regularizer=regularizers.l2(self.l2_reg)
            ),
            name='bidirectional_gru_3'
        )(x)
        x = LayerNormalization(name='layer_norm_final')(x)
        
        # Dual pooling for comprehensive feature extraction
        avg_pool = GlobalAveragePooling1D(name='global_avg_pool')(x)
        max_pool = GlobalMaxPooling1D(name='global_max_pool')(x)
        
        # Concatenate pooled features
        x = Concatenate(name='concat_pools')([avg_pool, max_pool])
        x = BatchNormalization(name='bn_pools')(x)
        x = Dropout(self.dropout_rate, name='dropout_pools')(x)
        
        # Dense layers
        for i, units in enumerate(self.dense_units):
            x = Dense(
                units=units,
                activation='relu',
                kernel_regularizer=regularizers.l2(self.l2_reg),
                name=f'dense_{i+1}'
            )(x)
            x = BatchNormalization(name=f'bn_dense_{i+1}')(x)
            x = Dropout(self.dropout_rate / 2, name=f'dropout_dense_{i+1}')(x)
        
        # Output layer
        outputs = Dense(1, activation='linear', name='output')(x)
        
        # Create model
        self.model = Model(inputs=inputs, outputs=outputs, name='Bidirectional_GRU')
        
        # Compile model
        self.model.compile(
            optimizer=tf.keras.optimizers.AdamW(
                learning_rate=self.learning_rate,
                weight_decay=self.l2_reg
            ),
            loss=tf.keras.losses.Huber(delta=1.0),
            metrics=['mae', 'mse']
        )
        
        return self.model
    
    def build_simple_model(self) -> Model:
        """
        Build a simpler GRU model for comparison.
        
        Returns:
            Compiled Keras Model
        """
        inputs = Input(shape=(self.sequence_length, self.n_features))
        
        x = Bidirectional(GRU(64, return_sequences=True))(inputs)
        x = Bidirectional(GRU(32, return_sequences=False))(x)
        x = Dense(16, activation='relu')(x)
        outputs = Dense(1)(x)
        
        model = Model(inputs=inputs, outputs=outputs, name='GRU_Simple')
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='huber',
            metrics=['mae', 'mse']
        )
        
        return model
    
    def summary(self):
        """Print model summary."""
        if self.model:
            self.model.summary()
        else:
            print("Model not built yet. Call build_model() first.")
    
    def get_model(self) -> Model:
        """Get the built model."""
        if self.model is None:
            self.build_model()
        return self.model
    
    @staticmethod
    def get_model_description() -> dict:
        """
        Get detailed model description for documentation.
        
        Returns:
            Dictionary with model information
        """
        return {
            'name': 'Bidirectional GRU with Residual Connections',
            'architecture': [
                'Input Layer (sequence_length, n_features)',
                'Temporal Convolution Block (multi-scale: 3, 5, 7)',
                'Bidirectional GRU (128 units) + Residual',
                'Layer Normalization',
                'Bidirectional GRU (64 units) + Residual',
                'Layer Normalization',
                'Bidirectional GRU (32 units)',
                'Global Average Pooling + Global Max Pooling',
                'Batch Normalization + Dropout',
                'Dense (128 units, ReLU)',
                'Dense (64 units, ReLU)',
                'Output (1 unit, Linear)'
            ],
            'advantages': [
                'Bidirectional processing captures context from both directions',
                'GRU is computationally more efficient than LSTM',
                'Residual connections enable training of deeper networks',
                'Multi-scale convolutions capture local patterns',
                'Dual pooling provides comprehensive feature aggregation',
                'Faster training compared to LSTM'
            ],
            'disadvantages': [
                'Bidirectional nature makes it unsuitable for strict real-time prediction',
                'Less memory than LSTM (no separate cell state)',
                'May underperform on very long sequences',
                'Increased model complexity from residual connections'
            ],
            'best_for': [
                'Medium-length sequences',
                'When training speed is important',
                'When bidirectional context is available',
                'Balanced accuracy-speed tradeoff scenarios'
            ]
        }


def create_bidirectional_gru_model(
    sequence_length: int = 30,
    n_features: int = 50,
    **kwargs
) -> Model:
    """
    Factory function to create Bidirectional GRU model.
    
    Args:
        sequence_length: Input sequence length
        n_features: Number of features
        **kwargs: Additional model parameters
        
    Returns:
        Compiled Keras Model
    """
    model_builder = BidirectionalGRUModel(
        sequence_length=sequence_length,
        n_features=n_features,
        **kwargs
    )
    return model_builder.build_model()
