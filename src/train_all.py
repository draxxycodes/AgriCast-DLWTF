"""
Comprehensive Model Training Script - MAXIMUM PARAMETERS VERSION

Trains ALL available models with maximum parameters and best architecture.
Creates comprehensive comparison charts with 12+ visualization types.
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
from datetime import datetime
import pickle
import json
import seaborn as sns

# TensorFlow setup
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print(f"✓ GPU: {gpus[0].name}")

import keras
from keras import layers, Model
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# =====================================================
# CONFIG - MAXIMUM PARAMETERS
# =====================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "daily_prices.csv"
MODEL_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
COMPARISON_DIR = FIGURES_DIR / "comparison"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"

SEQUENCE_LENGTH = 60
EPOCHS = 200
BATCH_SIZE = 32
PATIENCE = 35
LEARNING_RATE = 1e-4

# Model-specific learning rates (attention models need lower LR to prevent NaN)
MODEL_LR = {
    'Transformer': 5e-5,
    'Attention': 5e-5,
    'TFT': 8e-5,
    'default': 1e-4
}

def get_optimizer(model_name):
    """Get optimizer with gradient clipping for stability."""
    lr = MODEL_LR.get(model_name, MODEL_LR['default'])
    return keras.optimizers.AdamW(learning_rate=lr, clipnorm=1.0, weight_decay=1e-5)

# =====================================================
# DATA LOADING
# =====================================================
def load_data():
    """Load and prepare data."""
    print("\n📊 Loading data...")
    
    df = pd.read_csv(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    prices = df['price'].values
    features = pd.DataFrame()
    
    features['price'] = prices
    features['log_price'] = np.log1p(prices)
    features['pct_change'] = pd.Series(prices).pct_change().fillna(0)
    
    for w in [7, 14, 30]:
        features[f'ma_{w}'] = pd.Series(prices).rolling(w, min_periods=1).mean()
        features[f'std_{w}'] = pd.Series(prices).rolling(w, min_periods=1).std().fillna(0)
    
    features['momentum'] = prices - features['ma_7']
    
    scaler = RobustScaler()
    scaled = scaler.fit_transform(features.values)
    
    X, y = [], []
    for i in range(len(scaled) - SEQUENCE_LENGTH):
        X.append(scaled[i:i+SEQUENCE_LENGTH])
        y.append(scaled[i+SEQUENCE_LENGTH, 0])
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    
    n = len(X)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    
    splits = {
        'X_train': X[:train_end], 'y_train': y[:train_end],
        'X_val': X[train_end:val_end], 'y_val': y[train_end:val_end],
        'X_test': X[val_end:], 'y_test': y[val_end:],
        'dates': df['date'].values[SEQUENCE_LENGTH:],
        'prices': df['price'].values
    }
    
    print(f"✓ Features: {X.shape[2]}, Train: {splits['X_train'].shape[0]}")
    return splits, scaler


# =====================================================
# MODEL BUILDERS - MAXIMUM PARAMETERS
# =====================================================

def build_lstm(seq_len, n_feat):
    """LSTM with Attention - ~50M params."""
    inputs = layers.Input(shape=(seq_len, n_feat))
    x = layers.Dense(768)(inputs)
    x = layers.LayerNormalization()(x)
    
    # Deep Bidirectional LSTM stack
    x = layers.Bidirectional(layers.LSTM(768, return_sequences=True))(x)
    x = layers.Dropout(0.15)(x)
    x = layers.LayerNormalization()(x)
    
    x = layers.Bidirectional(layers.LSTM(640, return_sequences=True))(x)
    x = layers.Dropout(0.15)(x)
    x = layers.LayerNormalization()(x)
    
    x = layers.Bidirectional(layers.LSTM(512, return_sequences=True))(x)
    x = layers.Dropout(0.15)(x)
    x = layers.LayerNormalization()(x)
    
    # Multi-head attention
    attn = layers.MultiHeadAttention(num_heads=16, key_dim=64, dropout=0.1)(x, x)
    x = layers.Add()([x, attn])
    x = layers.LayerNormalization()(x)
    
    x = layers.Bidirectional(layers.LSTM(384, return_sequences=True))(x)
    x = layers.Dropout(0.15)(x)
    x = layers.LayerNormalization()(x)
    
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(x)
    attn2 = layers.MultiHeadAttention(num_heads=8, key_dim=48, dropout=0.1)(x, x)
    x = layers.Add()([x, attn2])
    x = layers.LayerNormalization()(x)
    
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(1024, activation='gelu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(512, activation='gelu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(256, activation='gelu')(x)
    x = layers.Dense(64, activation='gelu')(x)
    outputs = layers.Dense(1, dtype='float32')(x)
    model = Model(inputs, outputs, name='LSTM')
    model.compile(optimizer=get_optimizer('LSTM'), loss='huber', metrics=['mae'])
    return model


def build_gru(seq_len, n_feat):
    """GRU with Residual - ~60M params."""
    inputs = layers.Input(shape=(seq_len, n_feat))
    x = layers.Dense(1024)(inputs)
    x = layers.LayerNormalization()(x)
    
    # 8 deep residual GRU blocks
    for i in range(8):
        gru = layers.Bidirectional(layers.GRU(640, return_sequences=True))(x)
        gru = layers.Dropout(0.1)(gru)
        if gru.shape[-1] != x.shape[-1]:
            x = layers.Dense(gru.shape[-1])(x)
        x = layers.Add()([x, gru])
        x = layers.LayerNormalization()(x)
    
    # Attention layer with dropout
    attn = layers.MultiHeadAttention(num_heads=16, key_dim=80, dropout=0.1)(x, x)
    x = layers.Add()([x, attn])
    x = layers.LayerNormalization()(x)
    
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(1024, activation='gelu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(512, activation='gelu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(256, activation='gelu')(x)
    x = layers.Dense(128, activation='gelu')(x)
    outputs = layers.Dense(1, dtype='float32')(x)
    model = Model(inputs, outputs, name='GRU')
    model.compile(optimizer=get_optimizer('GRU'), loss='huber', metrics=['mae'])
    return model


def build_transformer(seq_len, n_feat):
    """Transformer Encoder - ~40M params (NaN-safe with pre-norm)."""
    inputs = layers.Input(shape=(seq_len, n_feat))
    
    # Input embedding with LayerNorm for stability
    x = layers.Dense(512)(inputs)
    x = layers.LayerNormalization()(x)
    
    # Learnable positional embedding
    positions = tf.range(start=0, limit=seq_len, delta=1)
    pos_emb = layers.Embedding(input_dim=seq_len, output_dim=512)(positions)
    x = x + pos_emb
    x = layers.Dropout(0.1)(x)
    
    # 12 Transformer blocks with PRE-NORM (critical for stability)
    for _ in range(12):
        # Pre-LayerNorm before attention (prevents gradient explosion)
        norm_x = layers.LayerNormalization()(x)
        attn = layers.MultiHeadAttention(num_heads=16, key_dim=64, dropout=0.1)(norm_x, norm_x)
        x = layers.Add()([x, attn])
        
        # Pre-LayerNorm before FFN
        norm_x = layers.LayerNormalization()(x)
        ffn = layers.Dense(2048, activation='gelu')(norm_x)
        ffn = layers.Dropout(0.1)(ffn)
        ffn = layers.Dense(512)(ffn)
        ffn = layers.Dropout(0.1)(ffn)
        x = layers.Add()([x, ffn])
    
    # Final LayerNorm
    x = layers.LayerNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    
    # Output head
    x = layers.Dense(512, activation='gelu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(256, activation='gelu')(x)
    x = layers.Dense(64, activation='gelu')(x)
    outputs = layers.Dense(1, dtype='float32')(x)
    model = Model(inputs, outputs, name='Transformer')
    model.compile(optimizer=get_optimizer('Transformer'), loss='huber', metrics=['mae'])
    return model


def build_tcn(seq_len, n_feat):
    """TCN - ~30M params."""
    inputs = layers.Input(shape=(seq_len, n_feat))
    x = layers.Conv1D(512, 1)(inputs)
    x = layers.LayerNormalization()(x)
    
    # Triple dilation stack with 512 filters
    for dilation in [1, 2, 4, 8, 16, 32, 1, 2, 4, 8, 16, 32, 1, 2, 4, 8, 16, 32]:
        conv = layers.Conv1D(512, 3, padding='causal', dilation_rate=dilation)(x)
        conv = layers.BatchNormalization()(conv)
        conv = layers.Activation('relu')(conv)
        conv = layers.Dropout(0.1)(conv)
        conv = layers.Conv1D(512, 3, padding='causal', dilation_rate=dilation)(conv)
        conv = layers.BatchNormalization()(conv)
        x = layers.Add()([x, conv])
        x = layers.Activation('relu')(x)
    
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(1024, activation='gelu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(512, activation='gelu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(256, activation='gelu')(x)
    x = layers.Dense(64, activation='gelu')(x)
    outputs = layers.Dense(1, dtype='float32')(x)
    model = Model(inputs, outputs, name='TCN')
    model.compile(optimizer=get_optimizer('TCN'), loss='huber', metrics=['mae'])
    return model


def build_wavenet(seq_len, n_feat):
    """WaveNet - ~30M params."""
    inputs = layers.Input(shape=(seq_len, n_feat))
    x = layers.Conv1D(512, 1)(inputs)
    x = layers.LayerNormalization()(x)
    skip_connections = []
    
    # Double stacks with wider filters
    for dilation in [1, 2, 4, 8, 16, 32, 1, 2, 4, 8, 16, 32, 1, 2, 4, 8, 16, 32, 1, 2, 4, 8]:
        if dilation > seq_len:
            dilation = seq_len // 2
        tanh_out = layers.Conv1D(512, 2, padding='causal', dilation_rate=dilation, activation='tanh')(x)
        sigm_out = layers.Conv1D(512, 2, padding='causal', dilation_rate=dilation, activation='sigmoid')(x)
        gated = layers.Multiply()([tanh_out, sigm_out])
        skip = layers.Conv1D(256, 1)(gated)
        skip_connections.append(skip)
        res = layers.Conv1D(512, 1)(gated)
        x = layers.Add()([x, res])
        x = layers.LayerNormalization()(x)
    
    x = layers.Add()(skip_connections)
    x = layers.Activation('relu')(x)
    x = layers.Conv1D(512, 1, activation='relu')(x)
    x = layers.Conv1D(256, 1, activation='relu')(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(512, activation='gelu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(256, activation='gelu')(x)
    x = layers.Dense(64, activation='gelu')(x)
    outputs = layers.Dense(1, dtype='float32')(x)
    model = Model(inputs, outputs, name='WaveNet')
    model.compile(optimizer=get_optimizer('WaveNet'), loss='huber', metrics=['mae'])
    return model


def build_nbeats(seq_len, n_feat):
    """N-BEATS - ~40M params."""
    inputs = layers.Input(shape=(seq_len, n_feat))
    x = layers.Flatten()(inputs)
    x = layers.LayerNormalization()(x)
    forecasts = []
    residual = x
    
    # 12 blocks with wider hidden layers
    for _ in range(12):
        h = layers.Dense(1024, activation='relu')(residual)
        h = layers.Dropout(0.1)(h)
        h = layers.LayerNormalization()(h)
        h = layers.Dense(1024, activation='relu')(h)
        h = layers.Dropout(0.1)(h)
        h = layers.Dense(512, activation='relu')(h)
        h = layers.Dense(256, activation='relu')(h)
        forecast = layers.Dense(1, dtype='float32')(h)
        forecasts.append(forecast)
        backcast = layers.Dense(seq_len * n_feat)(h)
        residual = layers.Subtract()([residual, backcast])
    
    outputs = layers.Add()(forecasts)
    model = Model(inputs, outputs, name='NBEATS')
    model.compile(optimizer=get_optimizer('NBEATS'), loss='huber', metrics=['mae'])
    return model


def build_tft(seq_len, n_feat):
    """Temporal Fusion Transformer - ~35M params."""
    inputs = layers.Input(shape=(seq_len, n_feat))
    
    # Variable selection with wider layers
    x = layers.Dense(512)(inputs)
    x = layers.LayerNormalization()(x)
    gate = layers.Dense(512, activation='sigmoid')(inputs)
    x = layers.Multiply()([x, gate])
    
    # Deep LSTM processing
    lstm_out = layers.Bidirectional(layers.LSTM(512, return_sequences=True))(x)
    lstm_out = layers.Dropout(0.1)(lstm_out)
    lstm_out = layers.LayerNormalization()(lstm_out)
    
    lstm_out = layers.Bidirectional(layers.LSTM(384, return_sequences=True))(lstm_out)
    lstm_out = layers.Dropout(0.1)(lstm_out)
    lstm_out = layers.LayerNormalization()(lstm_out)
    
    lstm_out = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(lstm_out)
    lstm_out = layers.LayerNormalization()(lstm_out)
    
    # Multi-head attention with dropout
    attn = layers.MultiHeadAttention(num_heads=16, key_dim=64, dropout=0.1)(lstm_out, lstm_out)
    x = layers.Add()([lstm_out, attn])
    x = layers.LayerNormalization()(x)
    
    # Second attention layer
    attn2 = layers.MultiHeadAttention(num_heads=8, key_dim=48, dropout=0.1)(x, x)
    x = layers.Add()([x, attn2])
    x = layers.LayerNormalization()(x)
    
    # Gated skip connection
    gate2 = layers.Dense(512, activation='sigmoid')(x)
    x = layers.Multiply()([x, gate2])
    
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(512, activation='gelu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(256, activation='gelu')(x)
    x = layers.Dense(64, activation='gelu')(x)
    outputs = layers.Dense(1, dtype='float32')(x)
    model = Model(inputs, outputs, name='TFT')
    model.compile(optimizer=get_optimizer('TFT'), loss='huber', metrics=['mae'])
    return model


def build_conv_lstm(seq_len, n_feat):
    """Conv-LSTM Hybrid - ~30M params."""
    inputs = layers.Input(shape=(seq_len, n_feat))
    
    # Deep convolutional feature extraction
    x = layers.Conv1D(256, 3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(384, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(512, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(512, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(384, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(256, 3, padding='same', activation='relu')(x)
    x = layers.LayerNormalization()(x)
    
    # Deep LSTM stack
    x = layers.Bidirectional(layers.LSTM(512, return_sequences=True))(x)
    x = layers.Dropout(0.15)(x)
    x = layers.LayerNormalization()(x)
    x = layers.Bidirectional(layers.LSTM(384, return_sequences=True))(x)
    x = layers.Dropout(0.15)(x)
    x = layers.LayerNormalization()(x)
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(x)
    x = layers.LayerNormalization()(x)
    x = layers.Bidirectional(layers.LSTM(128))(x)
    
    x = layers.Dense(512, activation='gelu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation='gelu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation='gelu')(x)
    x = layers.Dense(32, activation='gelu')(x)
    outputs = layers.Dense(1, dtype='float32')(x)
    model = Model(inputs, outputs, name='ConvLSTM')
    model.compile(optimizer=get_optimizer('ConvLSTM'), loss='huber', metrics=['mae'])
    return model


def build_dense(seq_len, n_feat):
    """Deep Dense Network - ~50M params."""
    inputs = layers.Input(shape=(seq_len, n_feat))
    x = layers.Flatten()(inputs)
    x = layers.LayerNormalization()(x)
    
    # Very deep and wide MLP
    for units in [2048, 2048, 1536, 1536, 1024, 1024, 512, 512, 256, 256, 128]:
        x = layers.Dense(units, activation='gelu')(x)
        x = layers.Dropout(0.15)(x)
        x = layers.BatchNormalization()(x)
    
    x = layers.Dense(64, activation='gelu')(x)
    outputs = layers.Dense(1, dtype='float32')(x)
    model = Model(inputs, outputs, name='DenseNN')
    model.compile(optimizer=get_optimizer('DenseNN'), loss='huber', metrics=['mae'])
    return model


def build_attention_only(seq_len, n_feat):
    """Pure Attention - ~35M params (NaN-safe with pre-norm)."""
    inputs = layers.Input(shape=(seq_len, n_feat))
    
    # Input embedding with normalization
    x = layers.Dense(384)(inputs)
    x = layers.LayerNormalization()(x)
    
    # Add positional encoding
    positions = tf.range(start=0, limit=seq_len, delta=1)
    pos_emb = layers.Embedding(input_dim=seq_len, output_dim=384)(positions)
    x = x + pos_emb
    x = layers.Dropout(0.1)(x)
    
    # 12 attention blocks with PRE-NORM (critical for stability)
    for _ in range(12):
        # Pre-LayerNorm before attention (prevents gradient explosion)
        norm_x = layers.LayerNormalization()(x)
        attn = layers.MultiHeadAttention(num_heads=12, key_dim=64, dropout=0.1)(norm_x, norm_x)
        x = layers.Add()([x, attn])
        
        # Pre-LayerNorm before FFN
        norm_x = layers.LayerNormalization()(x)
        ffn = layers.Dense(1536, activation='gelu')(norm_x)
        ffn = layers.Dropout(0.1)(ffn)
        ffn = layers.Dense(384)(ffn)
        ffn = layers.Dropout(0.1)(ffn)
        x = layers.Add()([x, ffn])
    
    # Final LayerNorm
    x = layers.LayerNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    
    x = layers.Dense(384, activation='gelu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation='gelu')(x)
    x = layers.Dense(64, activation='gelu')(x)
    outputs = layers.Dense(1, dtype='float32')(x)
    model = Model(inputs, outputs, name='Attention')
    model.compile(optimizer=get_optimizer('Attention'), loss='huber', metrics=['mae'])
    return model


# =====================================================
# INDIVIDUAL MODEL VISUALIZATION
# =====================================================

def plot_model_training(history, name, model_dir):
    """Plot training curves for a single model."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['loss']) + 1)
    
    axes[0].plot(epochs, history['loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r--', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss (Huber)', fontsize=12)
    axes[0].set_title(f'{name} - Training & Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')
    
    axes[1].plot(epochs, history['mae'], 'b-', label='Train MAE', linewidth=2)
    axes[1].plot(epochs, history['val_mae'], 'r--', label='Val MAE', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('MAE', fontsize=12)
    axes[1].set_title(f'{name} - Mean Absolute Error', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(model_dir / 'training_curves.png', dpi=200, bbox_inches='tight')
    plt.close()


def plot_model_predictions(y_true, y_pred, dates, name, model_dir):
    """Plot predictions vs actual for a single model."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    # Time series
    axes[0].plot(dates[-len(y_true):], y_true, 'b-', label='Actual', linewidth=1.5, alpha=0.8)
    axes[0].plot(dates[-len(y_pred):], y_pred, 'r--', label='Predicted', linewidth=1.5, alpha=0.7)
    axes[0].fill_between(dates[-len(y_true):], y_true, y_pred, alpha=0.2, color='gray')
    axes[0].set_xlabel('Date', fontsize=12)
    axes[0].set_ylabel('Price ($)', fontsize=12)
    axes[0].set_title(f'{name} - Actual vs Predicted Prices', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Scatter plot with regression line
    axes[1].scatter(y_true, y_pred, alpha=0.5, s=30, c='steelblue')
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    z = np.polyfit(y_true, y_pred, 1)
    p = np.poly1d(z)
    axes[1].plot([min_val, max_val], [p(min_val), p(max_val)], 'g-', linewidth=2, label=f'Regression (slope={z[0]:.3f})')
    axes[1].set_xlabel('Actual Price ($)', fontsize=12)
    axes[1].set_ylabel('Predicted Price ($)', fontsize=12)
    axes[1].set_title(f'{name} - Prediction Scatter Plot', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(model_dir / 'predictions.png', dpi=200, bbox_inches='tight')
    plt.close()


def plot_model_errors(errors, name, model_dir):
    """Plot error distribution for a single model."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Histogram
    axes[0].hist(errors, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].axvline(0, color='r', linestyle='--', linewidth=2)
    axes[0].axvline(np.mean(errors), color='g', linestyle='-', linewidth=2, label=f'Mean: {np.mean(errors):.2f}')
    axes[0].set_xlabel('Prediction Error ($)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title(f'{name} - Error Distribution', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Box plot by quartile
    abs_errors = np.abs(errors)
    q_data = [abs_errors[:len(abs_errors)//4], 
              abs_errors[len(abs_errors)//4:len(abs_errors)//2],
              abs_errors[len(abs_errors)//2:3*len(abs_errors)//4],
              abs_errors[3*len(abs_errors)//4:]]
    bp = axes[1].boxplot(q_data, patch_artist=True)
    colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    axes[1].set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4'])
    axes[1].set_xlabel('Time Period', fontsize=12)
    axes[1].set_ylabel('Absolute Error ($)', fontsize=12)
    axes[1].set_title(f'{name} - Error by Time Period', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    # Cumulative error
    sorted_errors = np.sort(abs_errors)
    cumulative = np.arange(1, len(sorted_errors)+1) / len(sorted_errors)
    axes[2].plot(sorted_errors, cumulative, 'b-', linewidth=2)
    axes[2].axhline(0.5, color='r', linestyle='--', alpha=0.5)
    axes[2].axhline(0.9, color='g', linestyle='--', alpha=0.5)
    axes[2].set_xlabel('Absolute Error ($)', fontsize=12)
    axes[2].set_ylabel('Cumulative Probability', fontsize=12)
    axes[2].set_title(f'{name} - Cumulative Error Distribution', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(model_dir / 'error_analysis.png', dpi=200, bbox_inches='tight')
    plt.close()


# =====================================================
# COMPREHENSIVE COMPARISON VISUALIZATION
# =====================================================

def plot_comprehensive_comparison(all_results, predictions_dict, errors_dict, histories_dict, y_true, dates):
    """Create comprehensive comparison plots - 12+ chart types."""
    results_df = pd.DataFrame(all_results).sort_values('rmse')
    
    print("  📊 Creating comparison charts...")
    
    # ===== 1. METRICS BAR CHARTS (4 metrics) =====
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    models = results_df['model'].values
    x = np.arange(len(models))
    
    # RMSE
    colors = ['#2ecc71' if v == results_df['rmse'].min() else '#3498db' for v in results_df['rmse']]
    axes[0,0].barh(x, results_df['rmse'], color=colors, edgecolor='black', alpha=0.8)
    axes[0,0].set_yticks(x)
    axes[0,0].set_yticklabels(models, fontsize=11)
    axes[0,0].set_xlabel('RMSE ($)', fontsize=12)
    axes[0,0].set_title('Root Mean Square Error (Lower = Better)', fontsize=14, fontweight='bold')
    axes[0,0].invert_yaxis()
    for i, v in enumerate(results_df['rmse']):
        axes[0,0].text(v + 5, i, f'{v:.1f}', va='center', fontsize=10)
    
    # MAE
    colors = ['#2ecc71' if v == results_df['mae'].min() else '#e74c3c' for v in results_df['mae']]
    axes[0,1].barh(x, results_df['mae'], color=colors, edgecolor='black', alpha=0.8)
    axes[0,1].set_yticks(x)
    axes[0,1].set_yticklabels(models, fontsize=11)
    axes[0,1].set_xlabel('MAE ($)', fontsize=12)
    axes[0,1].set_title('Mean Absolute Error (Lower = Better)', fontsize=14, fontweight='bold')
    axes[0,1].invert_yaxis()
    
    # R²
    colors = ['#2ecc71' if v == results_df['r2'].max() else '#9b59b6' for v in results_df['r2']]
    axes[1,0].barh(x, results_df['r2'], color=colors, edgecolor='black', alpha=0.8)
    axes[1,0].set_yticks(x)
    axes[1,0].set_yticklabels(models, fontsize=11)
    axes[1,0].set_xlabel('R² Score', fontsize=12)
    axes[1,0].set_title('R² Score (Higher = Better)', fontsize=14, fontweight='bold')
    axes[1,0].invert_yaxis()
    axes[1,0].axvline(0, color='red', linestyle='--', alpha=0.5)
    
    # MAPE
    colors = ['#2ecc71' if v == results_df['mape'].min() else '#f39c12' for v in results_df['mape']]
    axes[1,1].barh(x, results_df['mape'], color=colors, edgecolor='black', alpha=0.8)
    axes[1,1].set_yticks(x)
    axes[1,1].set_yticklabels(models, fontsize=11)
    axes[1,1].set_xlabel('MAPE (%)', fontsize=12)
    axes[1,1].set_title('Mean Absolute Percentage Error (Lower = Better)', fontsize=14, fontweight='bold')
    axes[1,1].invert_yaxis()
    
    plt.suptitle('Model Performance Metrics Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(COMPARISON_DIR / '01_metrics_bars.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("    ✓ Metrics bar charts")
    
    # ===== 2. PARAMETERS & EPOCHS =====
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    axes[0].barh(x, results_df['params'] / 1e6, color='coral', edgecolor='black', alpha=0.8)
    axes[0].set_yticks(x)
    axes[0].set_yticklabels(models, fontsize=11)
    axes[0].set_xlabel('Parameters (Millions)', fontsize=12)
    axes[0].set_title('Model Size Comparison', fontsize=14, fontweight='bold')
    axes[0].invert_yaxis()
    for i, v in enumerate(results_df['params']/1e6):
        axes[0].text(v + 0.2, i, f'{v:.1f}M', va='center', fontsize=10)
    
    axes[1].barh(x, results_df['epochs'], color='teal', edgecolor='black', alpha=0.8)
    axes[1].set_yticks(x)
    axes[1].set_yticklabels(models, fontsize=11)
    axes[1].set_xlabel('Training Epochs', fontsize=12)
    axes[1].set_title('Training Duration (Epochs until Early Stopping)', fontsize=14, fontweight='bold')
    axes[1].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(COMPARISON_DIR / '02_params_epochs.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("    ✓ Parameters & epochs")
    
    # ===== 3. PREDICTIONS OVERLAY =====
    fig, ax = plt.subplots(figsize=(20, 8))
    ax.plot(dates[-len(y_true):], y_true, 'k-', label='Actual', linewidth=2.5, alpha=0.9)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(predictions_dict)))
    for (name, pred), color in zip(predictions_dict.items(), colors):
        ax.plot(dates[-len(pred):], pred, '--', label=name, color=color, alpha=0.7, linewidth=1.2)
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price ($)', fontsize=12)
    ax.set_title('All Models - Predictions Comparison', fontsize=16, fontweight='bold')
    ax.legend(loc='upper left', ncol=4, fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(COMPARISON_DIR / '03_predictions_overlay.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("    ✓ Predictions overlay")
    
    # ===== 4. PERFORMANCE BUBBLE SCATTER =====
    fig, ax = plt.subplots(figsize=(14, 10))
    
    scatter = ax.scatter(results_df['rmse'], results_df['r2'], 
                        s=results_df['params']/30000, 
                        c=range(len(results_df)), cmap='viridis', alpha=0.7, edgecolors='black')
    
    for _, row in results_df.iterrows():
        ax.annotate(row['model'], (row['rmse'], row['r2']), 
                   xytext=(8, 8), textcoords='offset points', fontsize=11, fontweight='bold')
    
    ax.axhline(0, color='red', linestyle='--', alpha=0.5, label='R²=0 threshold')
    ax.set_xlabel('RMSE ($)', fontsize=14)
    ax.set_ylabel('R² Score', fontsize=14)
    ax.set_title('Model Performance Map (Bubble Size = Parameters)', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(COMPARISON_DIR / '04_performance_scatter.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("    ✓ Performance bubble scatter")
    
    # ===== 5. RADAR/SPIDER CHART =====
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
    
    # Normalize metrics for radar chart (0-1 scale, higher is better)
    metrics_norm = pd.DataFrame()
    metrics_norm['rmse'] = 1 - (results_df['rmse'] - results_df['rmse'].min()) / (results_df['rmse'].max() - results_df['rmse'].min() + 1e-8)
    metrics_norm['mae'] = 1 - (results_df['mae'] - results_df['mae'].min()) / (results_df['mae'].max() - results_df['mae'].min() + 1e-8)
    metrics_norm['mape'] = 1 - (results_df['mape'] - results_df['mape'].min()) / (results_df['mape'].max() - results_df['mape'].min() + 1e-8)
    metrics_norm['r2'] = (results_df['r2'] - results_df['r2'].min()) / (results_df['r2'].max() - results_df['r2'].min() + 1e-8)
    metrics_norm['efficiency'] = results_df['r2'] / (results_df['params'] / 1e6 + 0.1)
    metrics_norm['efficiency'] = (metrics_norm['efficiency'] - metrics_norm['efficiency'].min()) / (metrics_norm['efficiency'].max() - metrics_norm['efficiency'].min() + 1e-8)
    
    categories = ['RMSE\n(lower=better)', 'MAE\n(lower=better)', 'MAPE\n(lower=better)', 'R²\n(higher=better)', 'Efficiency\n(R²/params)']
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_df)))
    for idx, (_, row) in enumerate(results_df.iterrows()):
        values = [metrics_norm.iloc[idx]['rmse'], metrics_norm.iloc[idx]['mae'], 
                 metrics_norm.iloc[idx]['mape'], metrics_norm.iloc[idx]['r2'], metrics_norm.iloc[idx]['efficiency']]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=row['model'], color=colors[idx])
        ax.fill(angles, values, alpha=0.1, color=colors[idx])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_title('Model Performance Radar Chart', fontsize=16, fontweight='bold', y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=10)
    
    plt.tight_layout()
    plt.savefig(COMPARISON_DIR / '05_radar_chart.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("    ✓ Radar chart")
    
    # ===== 6. HEATMAP OF METRICS =====
    fig, ax = plt.subplots(figsize=(12, 10))
    
    heatmap_data = results_df[['model', 'rmse', 'mae', 'mape', 'r2']].copy()
    heatmap_data = heatmap_data.set_index('model')
    
    # Normalize for visualization
    heatmap_norm = heatmap_data.copy()
    for col in ['rmse', 'mae', 'mape']:
        heatmap_norm[col] = 1 - (heatmap_data[col] - heatmap_data[col].min()) / (heatmap_data[col].max() - heatmap_data[col].min() + 1e-8)
    heatmap_norm['r2'] = (heatmap_data['r2'] - heatmap_data['r2'].min()) / (heatmap_data['r2'].max() - heatmap_data['r2'].min() + 1e-8)
    
    sns.heatmap(heatmap_norm, annot=heatmap_data.round(2), fmt='', cmap='RdYlGn', 
                linewidths=0.5, ax=ax, cbar_kws={'label': 'Normalized Score (higher=better)'})
    ax.set_title('Model Performance Heatmap\n(Values = Original Metrics, Colors = Normalized)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Model', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(COMPARISON_DIR / '06_heatmap.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("    ✓ Heatmap")
    
    # ===== 7. BOX PLOTS OF ERRORS =====
    fig, ax = plt.subplots(figsize=(16, 8))
    
    error_data = [errors_dict[m] for m in results_df['model'].values if m in errors_dict]
    bp = ax.boxplot(error_data, patch_artist=True, labels=[m for m in results_df['model'].values if m in errors_dict])
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(error_data)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.axhline(0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Prediction Error ($)', fontsize=12)
    ax.set_title('Error Distribution by Model', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(COMPARISON_DIR / '07_error_boxplots.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("    ✓ Error box plots")
    
    # ===== 8. LEARNING CURVES COMPARISON =====
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(histories_dict)))
    for (name, hist), color in zip(histories_dict.items(), colors):
        epochs = range(1, len(hist['loss']) + 1)
        axes[0].plot(epochs, hist['loss'], '-', label=name, color=color, linewidth=1.5, alpha=0.8)
        axes[1].plot(epochs, hist['val_loss'], '-', label=name, color=color, linewidth=1.5, alpha=0.8)
    
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Training Loss (Huber)', fontsize=12)
    axes[0].set_title('Training Loss Curves', fontsize=14, fontweight='bold')
    axes[0].set_yscale('log')
    axes[0].legend(fontsize=9, loc='upper right')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Validation Loss (Huber)', fontsize=12)
    axes[1].set_title('Validation Loss Curves', fontsize=14, fontweight='bold')
    axes[1].set_yscale('log')
    axes[1].legend(fontsize=9, loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle('Learning Curves Comparison Across All Models', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(COMPARISON_DIR / '08_learning_curves.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("    ✓ Learning curves")
    
    # ===== 9. RESIDUAL PLOTS =====
    n_models = len(predictions_dict)
    cols = 4
    rows = (n_models + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5*rows))
    axes = axes.flatten()
    
    for idx, (name, pred) in enumerate(predictions_dict.items()):
        residuals = y_true - pred
        axes[idx].scatter(pred, residuals, alpha=0.5, s=15, c='steelblue')
        axes[idx].axhline(0, color='red', linestyle='--', linewidth=1.5)
        axes[idx].set_xlabel('Predicted', fontsize=10)
        axes[idx].set_ylabel('Residual', fontsize=10)
        axes[idx].set_title(f'{name}', fontsize=12, fontweight='bold')
        axes[idx].grid(True, alpha=0.3)
    
    for idx in range(len(predictions_dict), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Residual Plots (Predicted vs. Error) for All Models', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(COMPARISON_DIR / '09_residual_plots.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("    ✓ Residual plots")
    
    # ===== 10. CUMULATIVE ERROR DISTRIBUTION =====
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(errors_dict)))
    for (name, errs), color in zip(errors_dict.items(), colors):
        sorted_abs = np.sort(np.abs(errs))
        cumulative = np.arange(1, len(sorted_abs)+1) / len(sorted_abs)
        ax.plot(sorted_abs, cumulative, '-', label=name, color=color, linewidth=2, alpha=0.8)
    
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='50th percentile')
    ax.axhline(0.9, color='gray', linestyle=':', alpha=0.5, label='90th percentile')
    ax.set_xlabel('Absolute Error ($)', fontsize=12)
    ax.set_ylabel('Cumulative Probability', fontsize=12)
    ax.set_title('Cumulative Error Distribution - All Models', fontsize=16, fontweight='bold')
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(COMPARISON_DIR / '10_cumulative_error.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("    ✓ Cumulative error distribution")
    
    # ===== 11. EFFICIENCY PLOT (R² vs Parameters) =====
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(results_df)))
    scatter = ax.scatter(results_df['params']/1e6, results_df['r2'], 
                        s=200, c=results_df['rmse'], cmap='RdYlGn_r', 
                        edgecolors='black', alpha=0.8)
    
    for _, row in results_df.iterrows():
        ax.annotate(row['model'], (row['params']/1e6, row['r2']), 
                   xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')
    
    ax.axhline(0, color='red', linestyle='--', alpha=0.5)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('RMSE (Lower = Better)', fontsize=11)
    ax.set_xlabel('Parameters (Millions)', fontsize=12)
    ax.set_ylabel('R² Score', fontsize=12)
    ax.set_title('Model Efficiency: R² vs Model Size\n(Color = RMSE)', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(COMPARISON_DIR / '11_efficiency_plot.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("    ✓ Efficiency plot")
    
    # ===== 12. MAE vs RMSE SCATTER =====
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.scatter(results_df['mae'], results_df['rmse'], s=results_df['params']/30000, 
              c=results_df['r2'], cmap='coolwarm', edgecolors='black', alpha=0.8)
    
    for _, row in results_df.iterrows():
        ax.annotate(row['model'], (row['mae'], row['rmse']), 
                   xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')
    
    # Regression line
    z = np.polyfit(results_df['mae'], results_df['rmse'], 1)
    p = np.poly1d(z)
    ax.plot([results_df['mae'].min(), results_df['mae'].max()], 
            [p(results_df['mae'].min()), p(results_df['mae'].max())], 
            'g--', linewidth=2, label=f'Trend (slope={z[0]:.2f})')
    
    ax.set_xlabel('MAE ($)', fontsize=12)
    ax.set_ylabel('RMSE ($)', fontsize=12)
    ax.set_title('MAE vs RMSE Relationship\n(Size = Params, Color = R²)', fontsize=16, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(COMPARISON_DIR / '12_mae_vs_rmse.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("    ✓ MAE vs RMSE scatter")
    
    print("  ✓ All comparison plots saved!")


# =====================================================
# TRAINING
# =====================================================

def get_callbacks(name):
    return [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True, verbose=0),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=PATIENCE//3, min_lr=1e-7, verbose=0),
    ]


def evaluate(model, X_test, y_test, scaler):
    y_pred = model.predict(X_test, verbose=0).flatten()
    
    scale = scaler.scale_[0]
    center = scaler.center_[0]
    
    y_true_orig = y_test * scale + center
    y_pred_orig = y_pred * scale + center
    
    rmse = np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))
    mae = mean_absolute_error(y_true_orig, y_pred_orig)
    r2 = r2_score(y_true_orig, y_pred_orig)
    mape = np.mean(np.abs((y_true_orig - y_pred_orig) / (y_true_orig + 1e-8))) * 100
    
    return {
        'rmse': rmse, 'mae': mae, 'mape': mape, 'r2': r2,
        'y_pred': y_pred_orig, 'y_true': y_true_orig,
        'errors': y_true_orig - y_pred_orig
    }


def main():
    print("\n" + "=" * 70)
    print("🚀 COMPREHENSIVE MODEL TRAINING - MAXIMUM PARAMETERS")
    print("=" * 70)
    
    # Create directories
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    splits, scaler = load_data()
    seq_len = splits['X_train'].shape[1]
    n_feat = splits['X_train'].shape[2]
    dates = splits['dates']
    
    # All models to train
    MODELS = {
        'LSTM': build_lstm,
        'GRU': build_gru,
        'Transformer': build_transformer,
        'TCN': build_tcn,
        'WaveNet': build_wavenet,
        'NBEATS': build_nbeats,
        'TFT': build_tft,
        'ConvLSTM': build_conv_lstm,
        'DenseNN': build_dense,
        'Attention': build_attention_only,
    }
    
    all_results = []
    predictions_dict = {}
    errors_dict = {}
    histories_dict = {}
    y_true_final = None
    
    for i, (name, builder) in enumerate(MODELS.items(), 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(MODELS)}] 🔧 Training {name}")
        print('='*60)
        
        # Create model folder
        model_fig_dir = FIGURES_DIR / name.lower()
        model_fig_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            model = builder(seq_len, n_feat)
            param_count = model.count_params()
            print(f"   Parameters: {param_count:,} ({param_count/1e6:.2f}M)")
            
            history = model.fit(
                splits['X_train'], splits['y_train'],
                validation_data=(splits['X_val'], splits['y_val']),
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                callbacks=get_callbacks(name),
                verbose=1
            )
            
            # Evaluate
            metrics = evaluate(model, splits['X_test'], splits['y_test'], scaler)
            
            print(f"\n   ✓ RMSE: ${metrics['rmse']:.2f}")
            print(f"   ✓ MAE:  ${metrics['mae']:.2f}")
            print(f"   ✓ R²:   {metrics['r2']:.4f}")
            
            # Plot individual model figures
            plot_model_training(history.history, name, model_fig_dir)
            plot_model_predictions(metrics['y_true'], metrics['y_pred'], dates, name, model_fig_dir)
            plot_model_errors(metrics['errors'], name, model_fig_dir)
            print(f"   ✓ Figures saved to: {model_fig_dir}")
            
            # Save model
            model.save(MODEL_DIR / f"{name.lower()}.keras")
            
            # Store for comparison
            all_results.append({
                'model': name, 'rmse': metrics['rmse'], 'mae': metrics['mae'],
                'mape': metrics['mape'], 'r2': metrics['r2'],
                'params': param_count, 'epochs': len(history.history['loss'])
            })
            predictions_dict[name] = metrics['y_pred']
            errors_dict[name] = metrics['errors']
            histories_dict[name] = history.history
            y_true_final = metrics['y_true']
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)[:100]}")
            import traceback
            traceback.print_exc()
            continue
    
    # Generate comprehensive comparison plots
    print("\n📊 Generating comprehensive comparison plots...")
    if all_results and y_true_final is not None:
        plot_comprehensive_comparison(all_results, predictions_dict, errors_dict, histories_dict, y_true_final, dates)
    
    # Save results
    results_df = pd.DataFrame(all_results).sort_values('rmse')
    results_df.to_csv(REPORTS_DIR / "all_models_results.csv", index=False)
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 FINAL RESULTS")
    print("=" * 70)
    print("\n" + results_df[['model', 'rmse', 'mae', 'r2', 'params', 'epochs']].to_string(index=False))
    
    if len(results_df) > 0:
        best = results_df.iloc[0]
        print(f"\n🏆 BEST MODEL: {best['model']} ({best['params']/1e6:.1f}M params)")
        print(f"   RMSE: ${best['rmse']:.2f}, R²: {best['r2']:.4f}")
    
    print(f"\n📁 Figures: {FIGURES_DIR}")
    print(f"📁 Comparison: {COMPARISON_DIR}")
    print(f"📁 Models: {MODEL_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
