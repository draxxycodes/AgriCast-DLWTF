"""
LSTM Model with Self-Attention for Agricultural Price Prediction

This module implements a sophisticated LSTM architecture with:
- Multi-layer LSTM with increasing/decreasing units
- Self-attention mechanism for capturing important time steps
- Batch normalization and dropout for regularization
- Residual connections for better gradient flow
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, BatchNormalization,
    LayerNormalization, Concatenate, Multiply, Permute,
    RepeatVector, Flatten, Lambda, Add
)
import keras
import numpy as np
from typing import Tuple, Optional


@keras.saving.register_keras_serializable(package='models.lstm_model')
class AttentionLayer(layers.Layer):
    """
    Self-Attention mechanism for sequence data.
    
    Computes attention weights over time steps to focus
    on the most relevant historical information.
    """
    
    def __init__(
        self,
        units: int = 64,
        return_attention: bool = False,
        **kwargs
    ):
        super(AttentionLayer, self).__init__(**kwargs)
        self.units = units
        self.return_attention = return_attention
        
    def build(self, input_shape):
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
        self.u = self.add_weight(
            name='attention_context',
            shape=(self.units, 1),
            initializer='glorot_uniform',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, inputs, mask=None):
        # inputs shape: (batch_size, time_steps, features)
        
        # Compute attention scores
        score = tf.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        attention_weights = tf.nn.softmax(tf.tensordot(score, self.u, axes=1), axis=1)
        
        # Apply attention to get context vector
        context = inputs * attention_weights
        context = tf.reduce_sum(context, axis=1)
        
        if self.return_attention:
            return context, attention_weights
        return context
    
    def get_config(self):
        config = super(AttentionLayer, self).get_config()
        config.update({
            'units': self.units,
            'return_attention': self.return_attention
        })
        return config


@keras.saving.register_keras_serializable(package='models.lstm_model')
class MultiHeadAttention(layers.Layer):
    """
    Multi-Head Self-Attention for enhanced representation learning.
    """
    
    def __init__(
        self,
        num_heads: int = 4,
        key_dim: int = 32,
        dropout: float = 0.1,
        **kwargs
    ):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.dropout_rate = dropout
        
    def build(self, input_shape):
        self.mha = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.key_dim,
            dropout=self.dropout_rate
        )
        self.norm = LayerNormalization()
        self.dropout = Dropout(self.dropout_rate)
        super(MultiHeadAttention, self).build(input_shape)
    
    def call(self, inputs, training=None):
        attn_output = self.mha(inputs, inputs, training=training)
        attn_output = self.dropout(attn_output, training=training)
        return self.norm(inputs + attn_output)
    
    def get_config(self):
        config = super(MultiHeadAttention, self).get_config()
        config.update({
            'num_heads': self.num_heads,
            'key_dim': self.key_dim,
            'dropout': self.dropout_rate
        })
        return config


class LSTMAttentionModel:
    """
    Multi-layer LSTM with Self-Attention for time series forecasting.
    
    Architecture:
    - Input Layer
    - LSTM Layer 1 (128 units, return_sequences=True)
    - Batch Normalization + Dropout
    - LSTM Layer 2 (64 units, return_sequences=True)
    - Multi-Head Self-Attention
    - LSTM Layer 3 (32 units, return_sequences=False)
    - Dense layers with dropout
    - Output (1 unit for regression)
    
    Features:
    - Self-attention to focus on important time steps
    - Batch normalization for training stability
    - Dropout for regularization
    - L2 regularization on LSTM weights
    """
    
    def __init__(
        self,
        sequence_length: int = 30,
        n_features: int = 50,
        lstm_units: Tuple[int, ...] = (128, 64, 32),
        attention_units: int = 64,
        num_attention_heads: int = 4,
        dense_units: Tuple[int, ...] = (64, 32),
        dropout_rate: float = 0.3,
        l2_reg: float = 0.001,
        learning_rate: float = 0.001
    ):
        """
        Initialize LSTM with Attention model.
        
        Args:
            sequence_length: Number of time steps in input
            n_features: Number of input features
            lstm_units: Tuple of units for each LSTM layer
            attention_units: Units in attention layer
            num_attention_heads: Number of attention heads
            dense_units: Tuple of units for dense layers
            dropout_rate: Dropout probability
            l2_reg: L2 regularization factor
            learning_rate: Initial learning rate
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.attention_units = attention_units
        self.num_attention_heads = num_attention_heads
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.learning_rate = learning_rate
        
        self.model = None
        self.history = None
        
    def build_model(self) -> Model:
        """
        Build the LSTM with Attention architecture.
        
        Returns:
            Compiled Keras Model
        """
        # Input layer
        inputs = Input(shape=(self.sequence_length, self.n_features), name='input')
        
        # First LSTM layer
        x = LSTM(
            units=self.lstm_units[0],
            return_sequences=True,
            kernel_regularizer=regularizers.l2(self.l2_reg),
            recurrent_regularizer=regularizers.l2(self.l2_reg),
            name='lstm_1'
        )(inputs)
        x = BatchNormalization(name='bn_1')(x)
        x = Dropout(self.dropout_rate, name='dropout_1')(x)
        
        # Second LSTM layer
        x = LSTM(
            units=self.lstm_units[1],
            return_sequences=True,
            kernel_regularizer=regularizers.l2(self.l2_reg),
            recurrent_regularizer=regularizers.l2(self.l2_reg),
            name='lstm_2'
        )(x)
        x = BatchNormalization(name='bn_2')(x)
        x = Dropout(self.dropout_rate, name='dropout_2')(x)
        
        # Multi-Head Self-Attention
        attention_output = MultiHeadAttention(
            num_heads=self.num_attention_heads,
            key_dim=self.attention_units // self.num_attention_heads,
            dropout=self.dropout_rate,
            name='multi_head_attention'
        )(x)
        
        # Third LSTM layer
        x = LSTM(
            units=self.lstm_units[2],
            return_sequences=False,
            kernel_regularizer=regularizers.l2(self.l2_reg),
            recurrent_regularizer=regularizers.l2(self.l2_reg),
            name='lstm_3'
        )(attention_output)
        x = BatchNormalization(name='bn_3')(x)
        x = Dropout(self.dropout_rate, name='dropout_3')(x)
        
        # Dense layers
        for i, units in enumerate(self.dense_units):
            x = Dense(
                units=units,
                activation='relu',
                kernel_regularizer=regularizers.l2(self.l2_reg),
                name=f'dense_{i+1}'
            )(x)
            x = Dropout(self.dropout_rate / 2, name=f'dense_dropout_{i+1}')(x)
        
        # Output layer
        outputs = Dense(1, activation='linear', name='output')(x)
        
        # Create model
        self.model = Model(inputs=inputs, outputs=outputs, name='LSTM_Attention')
        
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
        Build a simpler LSTM model for comparison.
        
        Returns:
            Compiled Keras Model
        """
        inputs = Input(shape=(self.sequence_length, self.n_features))
        
        x = LSTM(64, return_sequences=True)(inputs)
        x = LSTM(32, return_sequences=False)(x)
        x = Dense(16, activation='relu')(x)
        outputs = Dense(1)(x)
        
        model = Model(inputs=inputs, outputs=outputs, name='LSTM_Simple')
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
            'name': 'LSTM with Multi-Head Self-Attention',
            'architecture': [
                'Input Layer (sequence_length, n_features)',
                'LSTM (128 units, return_sequences=True)',
                'Batch Normalization + Dropout (0.3)',
                'LSTM (64 units, return_sequences=True)',
                'Batch Normalization + Dropout (0.3)',
                'Multi-Head Self-Attention (4 heads)',
                'LSTM (32 units, return_sequences=False)',
                'Batch Normalization + Dropout (0.3)',
                'Dense (64 units, ReLU)',
                'Dense (32 units, ReLU)',
                'Output (1 unit, Linear)'
            ],
            'advantages': [
                'Captures long-term dependencies in sequential data',
                'Self-attention focuses on most relevant time steps',
                'Batch normalization stabilizes training',
                'Multi-head attention captures different aspects of patterns',
                'L2 regularization prevents overfitting'
            ],
            'disadvantages': [
                'Higher computational cost than vanilla LSTM',
                'Longer training time due to attention mechanism',
                'May require more data to train effectively',
                'Sequential nature limits parallelization'
            ],
            'best_for': [
                'Long sequences with important events at various points',
                'Data with complex temporal dependencies',
                'When interpretability of attention weights is valuable'
            ]
        }


def create_lstm_attention_model(
    sequence_length: int = 30,
    n_features: int = 50,
    **kwargs
) -> Model:
    """
    Factory function to create LSTM Attention model.
    
    Args:
        sequence_length: Input sequence length
        n_features: Number of features
        **kwargs: Additional model parameters
        
    Returns:
        Compiled Keras Model
    """
    model_builder = LSTMAttentionModel(
        sequence_length=sequence_length,
        n_features=n_features,
        **kwargs
    )
    return model_builder.build_model()
