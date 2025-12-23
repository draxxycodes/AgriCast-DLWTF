"""
Transformer Model for Time Series Forecasting

This module implements a Transformer architecture adapted for time series with:
- Sinusoidal positional encoding for temporal awareness
- Multi-head self-attention for capturing global dependencies
- Feed-forward networks with GELU activation
- Layer normalization and residual connections
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization,
    GlobalAveragePooling1D, Embedding, MultiHeadAttention,
    Add, BatchNormalization
)
import numpy as np
from typing import Tuple, Optional


class PositionalEncoding(layers.Layer):
    """
    Sinusoidal Positional Encoding for time series.
    
    Adds positional information to input embeddings using
    sine and cosine functions at different frequencies.
    """
    
    def __init__(self, max_length: int = 1000, d_model: int = 64, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.max_length = max_length
        self.d_model = d_model
        
    def build(self, input_shape):
        # Create positional encoding matrix
        position = np.arange(self.max_length)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.d_model, 2) * (-np.log(10000.0) / self.d_model))
        
        pe = np.zeros((self.max_length, self.d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        self.positional_encoding = tf.constant(pe[np.newaxis, :, :], dtype=tf.float32)
        
        super(PositionalEncoding, self).build(input_shape)
    
    def call(self, inputs):
        seq_length = tf.shape(inputs)[1]
        return inputs + self.positional_encoding[:, :seq_length, :]
    
    def get_config(self):
        config = super(PositionalEncoding, self).get_config()
        config.update({
            'max_length': self.max_length,
            'd_model': self.d_model
        })
        return config


class LearnablePositionalEncoding(layers.Layer):
    """
    Learnable Positional Encoding for time series.
    
    Uses trainable embeddings for positions instead of
    fixed sinusoidal patterns.
    """
    
    def __init__(self, max_length: int = 1000, d_model: int = 64, **kwargs):
        super(LearnablePositionalEncoding, self).__init__(**kwargs)
        self.max_length = max_length
        self.d_model = d_model
        
    def build(self, input_shape):
        self.position_embedding = self.add_weight(
            name='position_embedding',
            shape=(self.max_length, self.d_model),
            initializer='uniform',
            trainable=True
        )
        super(LearnablePositionalEncoding, self).build(input_shape)
    
    def call(self, inputs):
        seq_length = tf.shape(inputs)[1]
        positions = self.position_embedding[:seq_length, :]
        return inputs + positions
    
    def get_config(self):
        config = super(LearnablePositionalEncoding, self).get_config()
        config.update({
            'max_length': self.max_length,
            'd_model': self.d_model
        })
        return config


class TransformerBlock(layers.Layer):
    """
    Single Transformer encoder block.
    
    Components:
    - Multi-head self-attention
    - Position-wise feed-forward network
    - Layer normalization
    - Residual connections
    """
    
    def __init__(
        self,
        d_model: int = 64,
        num_heads: int = 8,
        ff_dim: int = 256,
        dropout_rate: float = 0.1,
        **kwargs
    ):
        super(TransformerBlock, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        
    def build(self, input_shape):
        self.attention = MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.d_model // self.num_heads,
            dropout=self.dropout_rate
        )
        self.ffn = tf.keras.Sequential([
            Dense(self.ff_dim, activation='gelu'),
            Dropout(self.dropout_rate),
            Dense(self.d_model),
            Dropout(self.dropout_rate)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(self.dropout_rate)
        self.dropout2 = Dropout(self.dropout_rate)
        
        super(TransformerBlock, self).build(input_shape)
    
    def call(self, inputs, training=None, mask=None):
        # Multi-head self-attention
        attn_output = self.attention(inputs, inputs, attention_mask=mask, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        # Feed-forward network
        ffn_output = self.ffn(out1, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2
    
    def get_config(self):
        config = super(TransformerBlock, self).get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'dropout_rate': self.dropout_rate
        })
        return config


class TransformerModel:
    """
    Transformer architecture for time series forecasting.
    
    Architecture:
    - Input projection layer
    - Positional Encoding (sinusoidal + learnable hybrid)
    - N Transformer Encoder Blocks
    - Global Average Pooling
    - Dense classification head
    - Output (1 unit for regression)
    
    Features:
    - Self-attention for capturing global dependencies
    - Parallel processing (unlike RNNs)
    - Positional encoding for temporal awareness
    - GELU activation for smooth gradients
    """
    
    def __init__(
        self,
        sequence_length: int = 30,
        n_features: int = 50,
        d_model: int = 64,
        num_heads: int = 8,
        num_layers: int = 4,
        ff_dim: int = 256,
        dense_units: Tuple[int, ...] = (64, 32),
        dropout_rate: float = 0.1,
        l2_reg: float = 0.001,
        learning_rate: float = 0.0001,
        use_learnable_pos: bool = True
    ):
        """
        Initialize Transformer model.
        
        Args:
            sequence_length: Number of time steps in input
            n_features: Number of input features
            d_model: Dimension of model embeddings
            num_heads: Number of attention heads
            num_layers: Number of transformer blocks
            ff_dim: Dimension of feed-forward network
            dense_units: Tuple of units for dense layers
            dropout_rate: Dropout probability
            l2_reg: L2 regularization factor
            learning_rate: Initial learning rate
            use_learnable_pos: Whether to use learnable positional encoding
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ff_dim = ff_dim
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.learning_rate = learning_rate
        self.use_learnable_pos = use_learnable_pos
        
        self.model = None
        self.history = None
        
    def build_model(self) -> Model:
        """
        Build the Transformer architecture.
        
        Returns:
            Compiled Keras Model
        """
        # Input layer
        inputs = Input(shape=(self.sequence_length, self.n_features), name='input')
        
        # Project input to d_model dimensions
        x = Dense(self.d_model, name='input_projection')(inputs)
        x = Dropout(self.dropout_rate, name='input_dropout')(x)
        
        # Add positional encoding
        if self.use_learnable_pos:
            x = LearnablePositionalEncoding(
                max_length=self.sequence_length,
                d_model=self.d_model,
                name='learnable_pos_encoding'
            )(x)
        else:
            x = PositionalEncoding(
                max_length=self.sequence_length,
                d_model=self.d_model,
                name='sinusoidal_pos_encoding'
            )(x)
        
        # Transformer encoder blocks
        for i in range(self.num_layers):
            x = TransformerBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                ff_dim=self.ff_dim,
                dropout_rate=self.dropout_rate,
                name=f'transformer_block_{i+1}'
            )(x)
        
        # Global pooling
        x = GlobalAveragePooling1D(name='global_avg_pool')(x)
        x = LayerNormalization(name='final_layer_norm')(x)
        
        # Dense classification head
        for i, units in enumerate(self.dense_units):
            x = Dense(
                units=units,
                activation='gelu',
                kernel_regularizer=regularizers.l2(self.l2_reg),
                name=f'dense_{i+1}'
            )(x)
            x = Dropout(self.dropout_rate / 2, name=f'dense_dropout_{i+1}')(x)
        
        # Output layer
        outputs = Dense(1, activation='linear', name='output')(x)
        
        # Create model
        self.model = Model(inputs=inputs, outputs=outputs, name='Transformer')
        
        # Custom learning rate schedule with warmup
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=self.learning_rate,
            decay_steps=10000,
            alpha=0.1
        )
        
        # Compile model
        self.model.compile(
            optimizer=tf.keras.optimizers.AdamW(
                learning_rate=lr_schedule,
                weight_decay=self.l2_reg,
                beta_1=0.9,
                beta_2=0.98,
                epsilon=1e-9
            ),
            loss=tf.keras.losses.Huber(delta=1.0),
            metrics=['mae', 'mse']
        )
        
        return self.model
    
    def build_simple_model(self) -> Model:
        """
        Build a simpler Transformer model for comparison.
        
        Returns:
            Compiled Keras Model
        """
        inputs = Input(shape=(self.sequence_length, self.n_features))
        
        x = Dense(32)(inputs)
        x = PositionalEncoding(self.sequence_length, 32)(x)
        x = TransformerBlock(32, 4, 64, 0.1)(x)
        x = TransformerBlock(32, 4, 64, 0.1)(x)
        x = GlobalAveragePooling1D()(x)
        x = Dense(16, activation='relu')(x)
        outputs = Dense(1)(x)
        
        model = Model(inputs=inputs, outputs=outputs, name='Transformer_Simple')
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
            'name': 'Transformer with Positional Encoding',
            'architecture': [
                'Input Layer (sequence_length, n_features)',
                'Input Projection (Dense to d_model)',
                'Positional Encoding (Learnable)',
                'Transformer Block 1 (8-head attention, FFN)',
                'Transformer Block 2 (8-head attention, FFN)',
                'Transformer Block 3 (8-head attention, FFN)',
                'Transformer Block 4 (8-head attention, FFN)',
                'Global Average Pooling',
                'Layer Normalization',
                'Dense (64 units, GELU)',
                'Dense (32 units, GELU)',
                'Output (1 unit, Linear)'
            ],
            'advantages': [
                'Captures long-range dependencies efficiently',
                'Parallel processing enables faster training',
                'Self-attention provides interpretable weights',
                'No vanishing gradient problem',
                'Scalable to longer sequences',
                'State-of-the-art performance on many tasks'
            ],
            'disadvantages': [
                'Higher memory usage (O(n²) for attention)',
                'May require more data to train effectively',
                'Computationally expensive for very long sequences',
                'Less inductive bias than RNNs for sequential data',
                'Requires careful hyperparameter tuning'
            ],
            'best_for': [
                'Long sequences with global dependencies',
                'When parallel training is important',
                'Large datasets where complexity is beneficial',
                'When interpretability of attention is valuable'
            ]
        }


def create_transformer_model(
    sequence_length: int = 30,
    n_features: int = 50,
    **kwargs
) -> Model:
    """
    Factory function to create Transformer model.
    
    Args:
        sequence_length: Input sequence length
        n_features: Number of features
        **kwargs: Additional model parameters
        
    Returns:
        Compiled Keras Model
    """
    model_builder = TransformerModel(
        sequence_length=sequence_length,
        n_features=n_features,
        **kwargs
    )
    return model_builder.build_model()
