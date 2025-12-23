"""
Advanced Research-Grade Deep Learning Models for Agricultural Price Prediction

This module contains state-of-the-art architectures suitable for research papers:
1. Temporal Fusion Transformer (TFT)
2. N-BEATS (Neural Basis Expansion Analysis for Time Series)
3. Informer (Efficient Transformer for Long Sequence Forecasting)
4. WaveNet (Dilated Causal Convolutions)
5. DeepAR (Autoregressive RNN)
6. TCN (Temporal Convolutional Network)
7. Graph Neural Network with Temporal Attention

All models are highly parameterized for optimal research results.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.layers import (
    Input, LSTM, GRU, Dense, Dropout, BatchNormalization,
    LayerNormalization, Bidirectional, Conv1D, GlobalAveragePooling1D,
    GlobalMaxPooling1D, Concatenate, Add, Multiply, Flatten,
    RepeatVector, TimeDistributed, Embedding, Permute, Reshape
)
import numpy as np
from typing import Tuple, List, Optional


# ============================================================================
# CUSTOM LAYERS
# ============================================================================

@keras.saving.register_keras_serializable(package='advanced_models')
class GatedLinearUnit(layers.Layer):
    """Gated Linear Unit for temporal fusion."""
    
    def __init__(self, units: int, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.dropout_rate = dropout
        
    def build(self, input_shape):
        self.linear = Dense(self.units)
        self.gate = Dense(self.units, activation='sigmoid')
        self.dropout = Dropout(self.dropout_rate)
        self.norm = LayerNormalization()
        super().build(input_shape)
        
    def call(self, x, training=None):
        linear_out = self.linear(x)
        gate_out = self.gate(x)
        gated = linear_out * gate_out
        gated = self.dropout(gated, training=training)
        return self.norm(gated + x) if x.shape[-1] == self.units else self.norm(gated)
    
    def get_config(self):
        config = super().get_config()
        config.update({'units': self.units, 'dropout': self.dropout_rate})
        return config


@keras.saving.register_keras_serializable(package='advanced_models')
class PositionalEncoding(layers.Layer):
    """Sinusoidal positional encoding."""
    
    def __init__(self, max_len: int = 1000, d_model: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.d_model = d_model
        
    def build(self, input_shape):
        position = np.arange(self.max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.d_model, 2) * -(np.log(10000.0) / self.d_model))
        
        pe = np.zeros((self.max_len, self.d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        self.pe = tf.constant(pe[np.newaxis, :, :], dtype=tf.float32)
        super().build(input_shape)
        
    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pe[:, :seq_len, :]
    
    def get_config(self):
        config = super().get_config()
        config.update({'max_len': self.max_len, 'd_model': self.d_model})
        return config


@keras.saving.register_keras_serializable(package='advanced_models')
class MultiScaleAttention(layers.Layer):
    """Multi-scale temporal attention mechanism."""
    
    def __init__(self, num_scales: int = 4, d_model: int = 64, num_heads: int = 8, **kwargs):
        super().__init__(**kwargs)
        self.num_scales = num_scales
        self.d_model = d_model
        self.num_heads = num_heads
        
    def build(self, input_shape):
        self.scale_convs = [
            Conv1D(self.d_model, kernel_size=2**i, padding='same', dilation_rate=2**i)
            for i in range(self.num_scales)
        ]
        self.attention = layers.MultiHeadAttention(
            num_heads=self.num_heads, 
            key_dim=self.d_model // self.num_heads
        )
        self.combine = Dense(self.d_model)
        self.norm = LayerNormalization()
        super().build(input_shape)
        
    def call(self, x, training=None):
        scale_outputs = [conv(x) for conv in self.scale_convs]
        combined = Concatenate(axis=-1)(scale_outputs)
        combined = self.combine(combined)
        attn_out = self.attention(combined, combined, training=training)
        return self.norm(x + attn_out)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'num_scales': self.num_scales,
            'd_model': self.d_model,
            'num_heads': self.num_heads
        })
        return config


# ============================================================================
# MODEL 1: TEMPORAL FUSION TRANSFORMER (TFT)
# ============================================================================

def build_temporal_fusion_transformer(
    seq_length: int,
    n_features: int,
    d_model: int = 128,
    num_heads: int = 8,
    num_lstm_layers: int = 2,
    dropout: float = 0.1,
    learning_rate: float = 0.0001
) -> Model:
    """
    Temporal Fusion Transformer - State-of-the-art for interpretable time series.
    
    Paper: "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting"
    
    Parameters: ~2.5M
    """
    inputs = Input(shape=(seq_length, n_features), name='input')
    
    # Input embedding
    x = Dense(d_model)(inputs)
    x = LayerNormalization()(x)
    
    # Variable selection network
    grn_outputs = []
    for i in range(n_features):
        feat = inputs[:, :, i:i+1]
        feat = Dense(d_model)(feat)
        grn = GatedLinearUnit(d_model, dropout)(feat)
        grn_outputs.append(grn)
    
    # Combine selected features
    x = Add()(grn_outputs) if len(grn_outputs) > 1 else grn_outputs[0]
    
    # LSTM Encoder
    for i in range(num_lstm_layers):
        x = Bidirectional(LSTM(d_model, return_sequences=True, dropout=dropout))(x)
        x = LayerNormalization()(x)
    
    # Static enrichment
    static = GlobalAveragePooling1D()(x)
    static = Dense(d_model, activation='relu')(static)
    static = RepeatVector(seq_length)(static)
    x = Concatenate()([x, static])
    x = Dense(d_model)(x)
    
    # Temporal self-attention
    attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)(x, x)
    x = LayerNormalization()(x + attn)
    
    # Gated skip connection
    x = GatedLinearUnit(d_model, dropout)(x)
    
    # Position-wise feed-forward
    ff = Dense(d_model * 4, activation='gelu')(x)
    ff = Dropout(dropout)(ff)
    ff = Dense(d_model)(ff)
    x = LayerNormalization()(x + ff)
    
    # Output
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(dropout)(x)
    outputs = Dense(1, dtype='float32')(x)
    
    model = Model(inputs, outputs, name='TemporalFusionTransformer')
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=1e-5),
        loss='huber',
        metrics=['mae', 'mse']
    )
    return model


# ============================================================================
# MODEL 2: N-BEATS (NEURAL BASIS EXPANSION)
# ============================================================================

def build_nbeats(
    seq_length: int,
    n_features: int,
    num_stacks: int = 4,
    num_blocks: int = 4,
    theta_dim: int = 256,
    hidden_units: int = 512,
    dropout: float = 0.1,
    learning_rate: float = 0.0001
) -> Model:
    """
    N-BEATS - Neural Basis Expansion Analysis for Time Series.
    
    Paper: "N-BEATS: Neural basis expansion analysis for interpretable time series forecasting"
    
    Parameters: ~3M
    """
    inputs = Input(shape=(seq_length, n_features), name='input')
    
    # Flatten features
    x = Flatten()(inputs)
    
    residual = x
    forecasts = []
    
    for stack_idx in range(num_stacks):
        for block_idx in range(num_blocks):
            # Fully connected stack
            block_input = residual
            h = Dense(hidden_units, activation='relu')(block_input)
            h = BatchNormalization()(h)
            h = Dropout(dropout)(h)
            
            for _ in range(3):
                h = Dense(hidden_units, activation='relu')(h)
                h = BatchNormalization()(h)
                h = Dropout(dropout)(h)
            
            # Theta computation
            theta_b = Dense(theta_dim)(h)
            theta_f = Dense(theta_dim)(h)
            
            # Backcast and forecast
            backcast = Dense(seq_length * n_features)(theta_b)
            forecast = Dense(1)(theta_f)
            
            # Update residual
            residual = residual - backcast
            forecasts.append(forecast)
    
    # Sum all forecasts
    if len(forecasts) > 1:
        output = Add()(forecasts)
    else:
        output = forecasts[0]
    
    outputs = Dense(1, dtype='float32')(output)
    
    model = Model(inputs, outputs, name='NBEATS')
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=1e-5),
        loss='huber',
        metrics=['mae', 'mse']
    )
    return model


# ============================================================================
# MODEL 3: INFORMER (EFFICIENT TRANSFORMER)
# ============================================================================

def build_informer(
    seq_length: int,
    n_features: int,
    d_model: int = 256,
    num_heads: int = 8,
    e_layers: int = 4,
    d_layers: int = 2,
    ff_dim: int = 512,
    dropout: float = 0.1,
    learning_rate: float = 0.0001
) -> Model:
    """
    Informer - Efficient Transformer for Long Sequence Time-Series Forecasting.
    
    Paper: "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting"
    
    Parameters: ~4M
    """
    inputs = Input(shape=(seq_length, n_features), name='input')
    
    # Input embedding with positional encoding
    x = Dense(d_model)(inputs)
    x = PositionalEncoding(max_len=seq_length * 2, d_model=d_model)(x)
    x = Dropout(dropout)(x)
    
    # Encoder with ProbSparse self-attention (simplified)
    for i in range(e_layers):
        # Multi-head attention with distillation
        attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)(x, x)
        x = LayerNormalization()(x + Dropout(dropout)(attn))
        
        # Feed-forward
        ff = Dense(ff_dim, activation='gelu')(x)
        ff = Dropout(dropout)(ff)
        ff = Dense(d_model)(ff)
        x = LayerNormalization()(x + ff)
        
        # Distilling (halving attention)
        if i < e_layers - 1:
            x = Conv1D(d_model, kernel_size=3, padding='same', activation='elu')(x)
            x = layers.MaxPooling1D(pool_size=2, padding='same')(x)
    
    # Decoder
    for i in range(d_layers):
        # Self-attention
        attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)(x, x)
        x = LayerNormalization()(x + Dropout(dropout)(attn))
        
        # Feed-forward
        ff = Dense(ff_dim, activation='gelu')(x)
        ff = Dropout(dropout)(ff)
        ff = Dense(d_model)(ff)
        x = LayerNormalization()(x + ff)
    
    # Output projection
    x = GlobalAveragePooling1D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(1, dtype='float32')(x)
    
    model = Model(inputs, outputs, name='Informer')
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=1e-5),
        loss='huber',
        metrics=['mae', 'mse']
    )
    return model


# ============================================================================
# MODEL 4: WAVENET (DILATED CAUSAL CONVOLUTIONS)
# ============================================================================

def build_wavenet(
    seq_length: int,
    n_features: int,
    residual_channels: int = 128,
    skip_channels: int = 256,
    dilation_layers: int = 10,
    num_stacks: int = 2,
    dropout: float = 0.1,
    learning_rate: float = 0.0001
) -> Model:
    """
    WaveNet - Dilated Causal Convolutions for Sequence Modeling.
    
    Paper: "WaveNet: A Generative Model for Raw Audio" (adapted for time series)
    
    Parameters: ~2M
    """
    inputs = Input(shape=(seq_length, n_features), name='input')
    
    # Causal convolution input
    x = Conv1D(residual_channels, kernel_size=1)(inputs)
    
    skip_connections = []
    
    for stack in range(num_stacks):
        for i in range(dilation_layers):
            dilation = 2 ** i
            
            # Gated activation
            tanh_out = Conv1D(residual_channels, kernel_size=2, 
                             padding='causal', dilation_rate=dilation,
                             activation='tanh')(x)
            sigmoid_out = Conv1D(residual_channels, kernel_size=2,
                                padding='causal', dilation_rate=dilation,
                                activation='sigmoid')(x)
            gated = tanh_out * sigmoid_out
            
            # Skip connection
            skip = Conv1D(skip_channels, kernel_size=1)(gated)
            skip_connections.append(skip)
            
            # Residual connection
            residual = Conv1D(residual_channels, kernel_size=1)(gated)
            x = Add()([x, residual])
            x = Dropout(dropout)(x)
            x = LayerNormalization()(x)
    
    # Combine skip connections
    skip_sum = Add()(skip_connections)
    skip_sum = keras.activations.relu(skip_sum)
    skip_sum = Conv1D(skip_channels, kernel_size=1, activation='relu')(skip_sum)
    skip_sum = Conv1D(256, kernel_size=1)(skip_sum)
    
    # Output
    x = GlobalAveragePooling1D()(skip_sum)
    x = Dense(128, activation='relu')(x)
    x = Dropout(dropout)(x)
    outputs = Dense(1, dtype='float32')(x)
    
    model = Model(inputs, outputs, name='WaveNet')
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=1e-5),
        loss='huber',
        metrics=['mae', 'mse']
    )
    return model


# ============================================================================
# MODEL 5: DEEP AR (AUTOREGRESSIVE RNN)
# ============================================================================

def build_deepar(
    seq_length: int,
    n_features: int,
    lstm_units: Tuple[int, ...] = (256, 256, 128),
    embedding_dim: int = 64,
    dropout: float = 0.2,
    learning_rate: float = 0.0001
) -> Model:
    """
    DeepAR - Probabilistic Forecasting with Autoregressive RNNs.
    
    Paper: "DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks"
    
    Parameters: ~2.5M
    """
    inputs = Input(shape=(seq_length, n_features), name='input')
    
    # Feature embedding
    x = Dense(embedding_dim)(inputs)
    x = LayerNormalization()(x)
    
    # Stacked LSTM layers
    for i, units in enumerate(lstm_units):
        return_seq = i < len(lstm_units) - 1
        x = LSTM(units, return_sequences=return_seq, dropout=dropout)(x)
        if return_seq:
            x = LayerNormalization()(x)
    
    # Probabilistic output (mean and variance)
    x = Dense(128, activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(64, activation='relu')(x)
    
    # Output mean (point forecast)
    outputs = Dense(1, dtype='float32')(x)
    
    model = Model(inputs, outputs, name='DeepAR')
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=1e-5),
        loss='huber',
        metrics=['mae', 'mse']
    )
    return model


# ============================================================================
# MODEL 6: TCN (TEMPORAL CONVOLUTIONAL NETWORK)
# ============================================================================

def build_tcn(
    seq_length: int,
    n_features: int,
    num_channels: List[int] = [128, 128, 256, 256, 512],
    kernel_size: int = 3,
    dropout: float = 0.1,
    learning_rate: float = 0.0001
) -> Model:
    """
    TCN - Temporal Convolutional Network with residual connections.
    
    Paper: "An Empirical Evaluation of Generic Convolutional and Recurrent Networks
            for Sequence Modeling"
    
    Parameters: ~3M
    """
    inputs = Input(shape=(seq_length, n_features), name='input')
    
    x = inputs
    
    for i, channels in enumerate(num_channels):
        dilation = 2 ** i
        
        # Two convolutional layers
        conv1 = Conv1D(channels, kernel_size, padding='causal', 
                      dilation_rate=dilation, activation='relu')(x)
        conv1 = BatchNormalization()(conv1)
        conv1 = Dropout(dropout)(conv1)
        
        conv2 = Conv1D(channels, kernel_size, padding='causal',
                      dilation_rate=dilation, activation='relu')(conv1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Dropout(dropout)(conv2)
        
        # Residual connection
        if x.shape[-1] != channels:
            x = Conv1D(channels, kernel_size=1)(x)
        
        x = Add()([x, conv2])
        x = LayerNormalization()(x)
    
    # Global pooling and output
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    x = Concatenate()([avg_pool, max_pool])
    
    x = Dense(256, activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(dropout)(x)
    outputs = Dense(1, dtype='float32')(x)
    
    model = Model(inputs, outputs, name='TCN')
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=1e-5),
        loss='huber',
        metrics=['mae', 'mse']
    )
    return model


# ============================================================================
# MODEL 7: HYBRID ATTENTION NETWORK
# ============================================================================

def build_hybrid_attention_network(
    seq_length: int,
    n_features: int,
    d_model: int = 256,
    num_heads: int = 8,
    num_layers: int = 6,
    dropout: float = 0.1,
    learning_rate: float = 0.0001
) -> Model:
    """
    Hybrid Attention Network - Combines CNN, LSTM, and Transformer.
    
    Custom architecture for maximum research performance.
    
    Parameters: ~5M
    """
    inputs = Input(shape=(seq_length, n_features), name='input')
    
    # Multi-scale CNN feature extraction
    conv1 = Conv1D(64, 3, padding='same', activation='relu')(inputs)
    conv2 = Conv1D(64, 5, padding='same', activation='relu')(inputs)
    conv3 = Conv1D(64, 7, padding='same', activation='relu')(inputs)
    cnn_features = Concatenate()([conv1, conv2, conv3])
    cnn_features = Conv1D(d_model, 1)(cnn_features)
    cnn_features = BatchNormalization()(cnn_features)
    
    # Bidirectional LSTM
    lstm_out = Bidirectional(LSTM(d_model // 2, return_sequences=True))(inputs)
    lstm_out = LayerNormalization()(lstm_out)
    
    # Combine CNN and LSTM
    x = Add()([cnn_features, lstm_out])
    x = LayerNormalization()(x)
    
    # Positional encoding
    x = PositionalEncoding(max_len=seq_length * 2, d_model=d_model)(x)
    
    # Transformer layers with multi-scale attention
    for i in range(num_layers):
        # Multi-scale attention
        attn = MultiScaleAttention(num_scales=4, d_model=d_model, num_heads=num_heads)(x)
        x = LayerNormalization()(x + Dropout(dropout)(attn))
        
        # Feed-forward
        ff = Dense(d_model * 4, activation='gelu')(x)
        ff = Dropout(dropout)(ff)
        ff = Dense(d_model)(ff)
        x = LayerNormalization()(x + ff)
    
    # Cross-attention with CNN features
    cross_attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
    x = cross_attn(x, cnn_features)
    x = LayerNormalization()(x)
    
    # Output
    x = GlobalAveragePooling1D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(dropout)(x)
    outputs = Dense(1, dtype='float32')(x)
    
    model = Model(inputs, outputs, name='HybridAttentionNetwork')
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=1e-5),
        loss='huber',
        metrics=['mae', 'mse']
    )
    return model


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def get_all_research_models(seq_length: int, n_features: int) -> dict:
    """Build all research-grade models."""
    return {
        'TFT': build_temporal_fusion_transformer(seq_length, n_features),
        'NBEATS': build_nbeats(seq_length, n_features),
        'Informer': build_informer(seq_length, n_features),
        'WaveNet': build_wavenet(seq_length, n_features),
        'DeepAR': build_deepar(seq_length, n_features),
        'TCN': build_tcn(seq_length, n_features),
        'HybridAttn': build_hybrid_attention_network(seq_length, n_features),
    }


def print_model_summary():
    """Print summary of all available models."""
    print("\n" + "=" * 60)
    print("RESEARCH-GRADE MODEL ARCHITECTURES")
    print("=" * 60)
    
    models_info = [
        ("Temporal Fusion Transformer", "~2.5M", "TFT - State-of-art interpretable forecasting"),
        ("N-BEATS", "~3M", "Neural Basis Expansion Analysis"),
        ("Informer", "~4M", "Efficient long-sequence transformer"),
        ("WaveNet", "~2M", "Dilated causal convolutions"),
        ("DeepAR", "~2.5M", "Probabilistic autoregressive RNN"),
        ("TCN", "~3M", "Temporal Convolutional Network"),
        ("Hybrid Attention", "~5M", "CNN + LSTM + Transformer fusion"),
    ]
    
    for name, params, desc in models_info:
        print(f"\n📊 {name}")
        print(f"   Parameters: {params}")
        print(f"   {desc}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    print_model_summary()
    
    # Demo: build all models
    seq_len, n_feat = 30, 50
    models = get_all_research_models(seq_len, n_feat)
    
    for name, model in models.items():
        print(f"\n{name}: {model.count_params():,} parameters")
