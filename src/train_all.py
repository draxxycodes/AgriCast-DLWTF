"""
Comprehensive Model Training Script - MAXIMUM PARAMETERS VERSION

Trains ALL available models with maximum parameters and best architecture.
Creates comprehensive comparison charts with 12+ visualization types.
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')
import argparse

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
# CONFIG - MAXIMUM PARAMETERS (FULL DATASET)
# =====================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed_agricultural.csv"  # CLEANED AGRICULTURAL DATA
MODEL_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
COMPARISON_DIR = FIGURES_DIR / "comparison"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"

SEQUENCE_LENGTH = 60
EPOCHS = 10  # Reduced to 30 for faster turnaround
BATCH_SIZE = 2048  # Increased to 2048 (16% usage -> Target 80%+)
PATIENCE = 5     # Aggressive early stopping
LEARNING_RATE = 1e-3  # Increased to 0.001 for faster convergence with Batch 2048

# Top commodities to one-hot encode (covers ~70% of data)
TOP_COMMODITIES = ['maize', 'wheat', 'rice', 'millet', 'sugar', 'sorghum', 
                   'wheat flour', 'potatoes', 'tomatoes', 'onions', 'lentils', 'beans (dry)']

# Model-specific learning rates (attention models need lower LR to prevent NaN)
MODEL_LR = {
    'Transformer': 1e-3,  # Increased from 1e-4
    'Attention': 1e-3,
    'TFT': 1e-3,
    'default': 1e-3       # Main speed boost
}

def get_optimizer(model_name):
    """Get optimizer with gradient clipping for stability."""
    lr = MODEL_LR.get(model_name, MODEL_LR['default'])
    return keras.optimizers.AdamW(learning_rate=lr, clipnorm=1.0, weight_decay=1e-5)

# Global storage for transformation parameters (needed for reverse transform)
TRANSFORM_PARAMS = {}

# =====================================================
# DATA LOADING - FULL DATASET WITH ENHANCED PREPROCESSING
# =====================================================
def load_data():
    """Load and prepare the preprocessed agricultural dataset.
    
    The data has already been cleaned and preprocessed by preprocess_clean.py:
    - Filtered to agricultural commodities only
    - Outliers removed via IQR
    - Log-transformed and normalized per commodity
    - Log-returns computed for stationarity
    """
    global TRANSFORM_PARAMS
    print("\n📊 Loading preprocessed agricultural dataset...")
    
    df = pd.read_csv(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date', 'price_normalized'])
    df = df.sort_values('date').reset_index(drop=True)
    
    print(f"  Records: {len(df):,}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Commodities: {df['commodity_clean'].nunique()}")
    
    # Select feature columns - EXCLUDE the target (price_normalized) to prevent leakage
    exclude_cols = ['date', 'price', 'price_log', 'price_normalized', 'commodity_clean', 
                    'commodity_mean', 'commodity_std', 'log_return']
    feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in ['float64', 'float32', 'int64', 'int32']]
    
    print(f"  Features: {len(feature_cols)}")
    
    # Prepare features
    features = df[feature_cols].copy()
    features = features.replace([np.inf, -np.inf], 0).fillna(0)
    
    # Scale features using RobustScaler
    scaler = RobustScaler()
    scaled = scaler.fit_transform(features.values).astype(np.float32)
    
    # Target: price_normalized (NOT log_return - those are nearly random!)
    # price_normalized is z-scored per commodity, comparable across commodities
    y = df['price_normalized'].values.astype(np.float32)
    
    # Store transform params
    TRANSFORM_PARAMS = {
        'target_type': 'normalized_price',
        'feature_cols': feature_cols,
    }
    
    # Create sequences
    print("  Creating sequences...")
    n_samples = len(scaled) - SEQUENCE_LENGTH
    n_features = scaled.shape[1]
    
    # Use strided view for memory efficiency
    X = np.lib.stride_tricks.sliding_window_view(scaled[:-1], (SEQUENCE_LENGTH, n_features))
    X = X[:n_samples, 0, :, :]
    y = y[SEQUENCE_LENGTH:n_samples + SEQUENCE_LENGTH]
    
    # Make contiguous copies
    X = np.ascontiguousarray(X, dtype=np.float32)
    y = np.ascontiguousarray(y, dtype=np.float32)
    
    # Train/Val/Test split (70/15/15)
    n = len(X)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    
    # Create tf.data.Datasets using generators
    print("  Creating tf.data.Datasets...")
    
    def make_generator(X_data, y_data):
        def gen():
            for i in range(len(X_data)):
                yield X_data[i], y_data[i]
        return gen
    
    output_signature = (
        tf.TensorSpec(shape=(SEQUENCE_LENGTH, n_features), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32)
    )
    
    train_ds = tf.data.Dataset.from_generator(
        make_generator(X_train, y_train),
        output_signature=output_signature
    ).shuffle(buffer_size=10000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    val_ds = tf.data.Dataset.from_generator(
        make_generator(X_val, y_val),
        output_signature=output_signature
    ).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    splits = {
        'train_ds': train_ds,
        'val_ds': val_ds,
        'X_test': X_test, 'y_test': y_test,
        'dates': df['date'].values[SEQUENCE_LENGTH:],
        'prices': df['price'].values,
        'n_train': len(X_train),
        'n_val': len(X_val),
        'n_test': len(X_test)
    }
    
    print(f"✓ Dataset ready!")
    print(f"  Features: {n_features}, Sequence length: {SEQUENCE_LENGTH}")
    print(f"  Train: {splits['n_train']:,} | Val: {splits['n_val']:,} | Test: {splits['n_test']:,}")
    return splits, scaler, n_features


# =====================================================
# MODEL BUILDERS - MAXIMUM PARAMETERS
# =====================================================

def build_lstm(seq_len, n_feat):
    """LSTM - Efficient Tiny Version (~2M params)."""
    inputs = layers.Input(shape=(seq_len, n_feat))
    x = layers.Dense(128)(inputs)
    # Block 1
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True, dropout=0.2))(x)
    x = layers.LayerNormalization()(x)
    # Block 2
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=False, dropout=0.2))(x)
    x = layers.LayerNormalization()(x)
    # Head
    x = layers.Dense(64, activation='gelu')(x)
    outputs = layers.Dense(1, dtype='float32')(x)
    model = Model(inputs, outputs, name='LSTM')
    model.compile(optimizer=get_optimizer('LSTM'), loss='huber', metrics=['mae'])
    return model


def build_gru(seq_len, n_feat):
    """GRU - Efficient Tiny Version (~2M params)."""
    inputs = layers.Input(shape=(seq_len, n_feat))
    x = layers.Dense(128)(inputs)
    # Block 1
    x = layers.Bidirectional(layers.GRU(256, return_sequences=True, dropout=0.2))(x)
    x = layers.LayerNormalization()(x)
    # Block 2
    x = layers.Bidirectional(layers.GRU(128, return_sequences=False, dropout=0.2))(x)
    x = layers.LayerNormalization()(x)
    # Head
    x = layers.Dense(64, activation='gelu')(x)
    outputs = layers.Dense(1, dtype='float32')(x)
    model = Model(inputs, outputs, name='GRU')
    model.compile(optimizer=get_optimizer('GRU'), loss='huber', metrics=['mae'])
    return model


def build_transformer(seq_len, n_feat):
    """Transformer - Efficient Tiny Version (~2M params)."""
    d_model = 128
    num_heads = 4
    ff_dim = 512
    num_blocks = 4
    
    inputs = layers.Input(shape=(seq_len, n_feat))
    x = layers.Dense(d_model)(inputs)
    # Positional Embedding
    pos_emb = layers.Embedding(input_dim=seq_len, output_dim=d_model)(tf.range(start=0, limit=seq_len, delta=1))
    x = x + pos_emb
    x = layers.Dropout(0.1)(x)
    
    for _ in range(num_blocks):
        # Attention
        norm_x = layers.LayerNormalization()(x)
        attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model//num_heads, dropout=0.1)(norm_x, norm_x)
        x = layers.Add()([x, attn])
        # FFN
        norm_x = layers.LayerNormalization()(x)
        ffn = layers.Dense(ff_dim, activation='gelu')(norm_x)
        ffn = layers.Dropout(0.1)(ffn)
        ffn = layers.Dense(d_model)(ffn)
        x = layers.Add()([x, ffn])
        
    x = layers.LayerNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='gelu')(x)
    outputs = layers.Dense(1, dtype='float32')(x)
    model = Model(inputs, outputs, name='Transformer')
    model.compile(optimizer=get_optimizer('Transformer'), loss='huber', metrics=['mae'])
    return model


def build_tcn(seq_len, n_feat):
    """TCN - Efficient Tiny Version (~1M params)."""
    filters = 128
    kernel_size = 3
    dilations = [1, 2, 4, 8, 16]
    
    inputs = layers.Input(shape=(seq_len, n_feat))
    x = layers.Conv1D(filters, 1)(inputs)
    
    for dilation in dilations:
        # Residual Block
        res = x
        x = layers.Conv1D(filters, kernel_size, padding='causal', dilation_rate=dilation, activation='gelu')(x)
        x = layers.SpatialDropout1D(0.1)(x)
        x = layers.Conv1D(filters, kernel_size, padding='causal', dilation_rate=dilation, activation='gelu')(x)
        x = layers.Add()([x, res])
        x = layers.LayerNormalization()(x)
        
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='gelu')(x)
    outputs = layers.Dense(1, dtype='float32')(x)
    model = Model(inputs, outputs, name='TCN')
    model.compile(optimizer=get_optimizer('TCN'), loss='huber', metrics=['mae'])
    return model


def build_wavenet(seq_len, n_feat):
    """WaveNet - Efficient Tiny Version (~1M params)."""
    filters = 128
    dilations = [1, 2, 4, 8, 16, 32]
    
    inputs = layers.Input(shape=(seq_len, n_feat))
    x = layers.Conv1D(filters, 1)(inputs)
    skip_connections = []
    
    for dilation in dilations:
        # Gated Activation
        tanh_out = layers.Conv1D(filters, 2, padding='causal', dilation_rate=dilation, activation='tanh')(x)
        sigm_out = layers.Conv1D(filters, 2, padding='causal', dilation_rate=dilation, activation='sigmoid')(x)
        gated = layers.Multiply()([tanh_out, sigm_out])
        
        # Skip & Residual
        skip = layers.Conv1D(filters, 1)(gated)
        skip_connections.append(skip)
        x = layers.Add()([x, layers.Conv1D(filters, 1)(gated)])
        x = layers.LayerNormalization()(x)
        
    x = layers.Add()(skip_connections)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='gelu')(x)
    outputs = layers.Dense(1, dtype='float32')(x)
    model = Model(inputs, outputs, name='WaveNet')
    model.compile(optimizer=get_optimizer('WaveNet'), loss='huber', metrics=['mae'])
    return model


def build_nbeats(seq_len, n_feat):
    """N-BEATS Optimized for Time Series Forecasting.
    
    Key optimizations:
    - Interpretable architecture (trend + seasonality blocks)
    - GELU activation for smoother gradients
    - Proper regularization to prevent overfitting
    - Residual stacking for deep learning
    - Shared weights within stack for efficiency
    
    Target: ~1.2M params (Tiny Version for Efficiency)
    """
    num_stacks = 2
    num_blocks_per_stack = 3  # Reduced from 4
    hidden_dim = 128          # Reduced from 512
    theta_dim = 128           # Reduced from 256
    
    inputs = layers.Input(shape=(seq_len, n_feat))
    x = layers.Flatten()(inputs)
    x = layers.LayerNormalization()(x)
    
    block_input_dim = seq_len * n_feat
    forecasts = []
    residual = x
    
    for stack_idx in range(num_stacks):
        for block_idx in range(num_blocks_per_stack):
            # Graduated dropout per block
            dropout_rate = 0.1 + (stack_idx * 0.05) + (block_idx * 0.02)
            
            # FC stack with proper regularization
            h = layers.Dense(hidden_dim, activation='gelu',
                           kernel_regularizer=keras.regularizers.l2(1e-5))(residual)
            h = layers.Dropout(dropout_rate)(h)
            h = layers.LayerNormalization()(h)
            
            h = layers.Dense(hidden_dim, activation='gelu',
                           kernel_regularizer=keras.regularizers.l2(1e-5))(h)
            h = layers.Dropout(dropout_rate)(h)
            h = layers.LayerNormalization()(h)
            
            h = layers.Dense(hidden_dim, activation='gelu',
                           kernel_regularizer=keras.regularizers.l2(1e-5))(h)
            h = layers.Dropout(dropout_rate * 0.5)(h)
            
            h = layers.Dense(theta_dim, activation='gelu',
                           kernel_regularizer=keras.regularizers.l2(1e-5))(h)
            
            # Forecast output
            forecast = layers.Dense(1, dtype='float32')(h)
            forecasts.append(forecast)
            
            # Backcast for residual update
            backcast = layers.Dense(block_input_dim,
                                   kernel_regularizer=keras.regularizers.l2(1e-5))(h)
            
            # Update residual
            residual = layers.Subtract()([residual, backcast])
            residual = layers.LayerNormalization()(residual)
    
    # Aggregate all forecasts
    outputs = layers.Add()(forecasts)
    
    model = Model(inputs, outputs, name='NBEATS')
    model.compile(optimizer=get_optimizer('NBEATS'), loss='huber', metrics=['mae'])
    return model


def build_patchtst(seq_len, n_feat):
    """PatchTST - Efficient Tiny Version (~1M params)."""
    d_model = 128
    num_heads = 4
    ff_dim = 256
    num_blocks = 2
    patch_len = 16
    stride = 8
    
    inputs = layers.Input(shape=(seq_len, n_feat))
    
    # 1. Instance Norm (RevIN concept)
    mean = layers.Reshape((1, n_feat))(layers.AveragePooling1D(pool_size=seq_len)(inputs))
    std = layers.Reshape((1, n_feat))(layers.AveragePooling1D(pool_size=seq_len)(layers.Subtract()([inputs, mean])**2))
    std = layers.Activation(lambda x: tf.sqrt(x + 1e-5))(std)
    x = layers.Subtract()([inputs, mean])
    # logical division using Lambda
    x = layers.Lambda(lambda args: args[0] / (args[1] + 1e-5))([x, std])
    
    # 2. Patching (Conv1D)
    x = layers.Conv1D(d_model, patch_len, strides=stride, padding='valid')(x)
    x = layers.LayerNormalization()(x)
    
    # 3. Transformer Backbone
    for _ in range(num_blocks):
        # Attention
        norm_x = layers.LayerNormalization()(x)
        attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model//num_heads, dropout=0.1)(norm_x, norm_x)
        x = layers.Add()([x, attn])
        # FFN
        norm_x = layers.LayerNormalization()(x)
        ffn = layers.Dense(ff_dim, activation='gelu')(norm_x)
        ffn = layers.Dropout(0.1)(ffn)
        ffn = layers.Dense(d_model)(ffn)
        x = layers.Add()([x, ffn])
    
    x = layers.LayerNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    
    # 4. Head
    x = layers.Dense(64, activation='gelu')(x)
    outputs = layers.Dense(1, dtype='float32')(x)
    
    # Simple Denorm (optional, usually handled by target scaling, but we output normalized)
    
    model = Model(inputs, outputs, name='PatchTST')
    model.compile(optimizer=get_optimizer('PatchTST'), loss='huber', metrics=['mae'])
    return model


# =====================================================
# MODEL REGISTRY
# =====================================================
MODEL_BUILDERS = {
    'LSTM': build_lstm,
    'GRU': build_gru,
    'Transformer': build_transformer,
    'TCN': build_tcn,
    'WaveNet': build_wavenet,
    'NBEATS': build_nbeats,
    'PatchTST': build_patchtst
}

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
    
    # Downsample if too many points to avoid matplotlib overflow
    MAX_POINTS = 5000
    if len(y_true) > MAX_POINTS:
        step = len(y_true) // MAX_POINTS
        y_true_plot = y_true[::step]
        y_pred_plot = y_pred[::step]
        dates_plot = dates[-len(y_true):][::step]
    else:
        y_true_plot = y_true
        y_pred_plot = y_pred
        dates_plot = dates[-len(y_true):]
    
    # Time series
    axes[0].plot(dates_plot, y_true_plot, 'b-', label='Actual', linewidth=1.5, alpha=0.8)
    axes[0].plot(dates_plot, y_pred_plot, 'r--', label='Predicted', linewidth=1.5, alpha=0.7)
    axes[0].fill_between(dates_plot, y_true_plot, y_pred_plot, alpha=0.2, color='gray')
    axes[0].set_xlabel('Date', fontsize=12)
    axes[0].set_ylabel('Price ($)', fontsize=12)
    axes[0].set_title(f'{name} - Actual vs Predicted Prices', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Scatter plot with regression line (use downsampled data)
    axes[1].scatter(y_true_plot, y_pred_plot, alpha=0.5, s=30, c='steelblue')
    min_val = min(y_true.min(), y_pred.min())  # Use full data for regression fit
    max_val = max(y_true.max(), y_pred.max())
    axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    # Cast to float64 to fix numpy linalg float16 error
    z = np.polyfit(y_true.astype(np.float64), y_pred.astype(np.float64), 1)
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
    
    # Downsample if too many points for histogram/cumulative plots
    MAX_POINTS = 10000
    if len(errors) > MAX_POINTS:
        step = len(errors) // MAX_POINTS
        errors_sample = errors[::step]
    else:
        errors_sample = errors
    
    # Histogram (use sampled data)
    axes[0].hist(errors_sample, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].axvline(0, color='r', linestyle='--', linewidth=2)
    axes[0].axvline(np.mean(errors), color='g', linestyle='-', linewidth=2, label=f'Mean: {np.mean(errors):.2f}')
    axes[0].set_xlabel('Prediction Error ($)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title(f'{name} - Error Distribution', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Box plot by quartile (use sampled data)
    abs_errors = np.abs(errors_sample)
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
    
    # Cumulative error (use sampled data)
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
    
    # Downsample for plotting if too many points
    MAX_POINTS = 5000
    if len(y_true) > MAX_POINTS:
        step = len(y_true) // MAX_POINTS
        y_true_plot = y_true[::step]
        dates_plot = dates[-len(y_true):][::step]
    else:
        y_true_plot = y_true
        dates_plot = dates[-len(y_true):]
    
    ax.plot(dates_plot, y_true_plot, 'k-', label='Actual', linewidth=2.5, alpha=0.9)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(predictions_dict)))
    for (name, pred), color in zip(predictions_dict.items(), colors):
        if len(pred) > MAX_POINTS:
            pred_plot = pred[::step]
        else:
            pred_plot = pred
        ax.plot(dates_plot[:len(pred_plot)], pred_plot, '--', label=name, color=color, alpha=0.7, linewidth=1.2)
    
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
    
    # Downsample for scatter plots
    MAX_POINTS = 3000
    step = max(1, len(y_true) // MAX_POINTS)
    
    for idx, (name, pred) in enumerate(predictions_dict.items()):
        residuals = y_true - pred
        # Use downsampled data for scatter
        axes[idx].scatter(pred[::step], residuals[::step], alpha=0.5, s=15, c='steelblue')
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
        # OneCycleLR-style: reduce LR faster for quick convergence
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, min_lr=1e-7, verbose=0),
    ]


def evaluate(model, X_test, y_test, scaler):
    """Evaluate model with comprehensive metrics including directional accuracy."""
    y_pred = model.predict(X_test, verbose=0).flatten()
    
    # For log-returns, we evaluate on the normalized values directly
    # since they represent returns, not absolute prices
    y_true = y_test
    y_pred_vals = y_pred
    
    # Basic metrics on normalized log-returns
    rmse = np.sqrt(mean_squared_error(y_true, y_pred_vals))
    mae = mean_absolute_error(y_true, y_pred_vals)
    r2 = r2_score(y_true, y_pred_vals)
    
    # ===== NEW METRICS FOR LOG-RETURNS =====
    # Directional Accuracy: % of times we correctly predict the sign (up/down)
    actual_direction = np.sign(y_true)
    pred_direction = np.sign(y_pred_vals)
    directional_accuracy = np.mean(actual_direction == pred_direction) * 100
    
    # Information Coefficient (IC): Spearman rank correlation
    # Measures if we rank returns correctly, even if magnitudes are off
    from scipy.stats import spearmanr
    ic, ic_pvalue = spearmanr(y_true, y_pred_vals)
    
    # MAPE (less meaningful for returns, but kept for compatibility)
    mape = np.mean(np.abs((y_true - y_pred_vals) / (np.abs(y_true) + 1e-8))) * 100
    mape = min(mape, 999.9)  # Cap extreme values
    
    return {
        'rmse': rmse, 'mae': mae, 'mape': mape, 'r2': r2,
        'directional_acc': directional_accuracy,
        'ic': ic, 'ic_pvalue': ic_pvalue,
        'y_pred': y_pred_vals, 'y_true': y_true,
        'errors': y_true - y_pred_vals
    }


def train_single_model(name, splits, scaler, seq_len, n_feat, dates):
    """Train a single model and generate its figures."""
    # 7 core architectures for comprehensive time series forecasting
    MODELS = {
        'LSTM': build_lstm,
        'GRU': build_gru,
        'Transformer': build_transformer,
        'TCN': build_tcn,
        'WaveNet': build_wavenet,
        'NBEATS': build_nbeats,
        'PatchTST': build_patchtst,
    }
    
    if name not in MODELS:
        print(f"❌ Unknown model: {name}")
        print(f"   Available: {', '.join(MODELS.keys())}")
        return None
    
    builder = MODELS[name]
    model_fig_dir = FIGURES_DIR / name.lower()
    model_fig_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"🔧 Training {name}")
    print('='*60)
    
    try:
        model = builder(seq_len, n_feat)
        param_count = model.count_params()
        print(f"   Parameters: {param_count:,} ({param_count/1e6:.2f}M)")
        
        history = model.fit(
            splits['train_ds'],
            validation_data=splits['val_ds'],
            epochs=EPOCHS,
            callbacks=get_callbacks(name),
            verbose=1
        )
        
        metrics = evaluate(model, splits['X_test'], splits['y_test'], scaler)
        
        # Print metrics (now for log-returns, not prices)
        print(f"\n   ✓ RMSE: {metrics['rmse']:.4f}")
        print(f"   ✓ MAE:  {metrics['mae']:.4f}")
        print(f"   ✓ R²:   {metrics['r2']:.4f}")
        print(f"   ✓ Directional Accuracy: {metrics['directional_acc']:.1f}%")
        print(f"   ✓ Information Coef (IC): {metrics['ic']:.4f}")
        
        # Plot individual model figures
        plot_model_training(history.history, name, model_fig_dir)
        plot_model_predictions(metrics['y_true'], metrics['y_pred'], dates, name, model_fig_dir)
        plot_model_errors(metrics['errors'], name, model_fig_dir)
        print(f"   ✓ Figures saved to: {model_fig_dir}")
        
        # Save model
        model.save(MODEL_DIR / f"{name.lower()}.keras")
        print(f"   ✓ Model saved to: {MODEL_DIR / f'{name.lower()}.keras'}")
        
        # Save individual results (including new metrics)
        result = {
            'model': name, 'rmse': metrics['rmse'], 'mae': metrics['mae'],
            'mape': metrics['mape'], 'r2': metrics['r2'],
            'directional_acc': metrics['directional_acc'],
            'ic': metrics['ic'], 'ic_pvalue': metrics['ic_pvalue'],
            'params': param_count, 'epochs': len(history.history['loss'])
        }
        result_file = REPORTS_DIR / f"{name.lower()}_results.json"
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        return result
        
    except Exception as e:
        print(f"   ❌ Error: {str(e)[:100]}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        keras.backend.clear_session()
        import gc
        gc.collect()
        tf.keras.backend.clear_session()


def generate_comparison():
    """Generate comparison charts for all trained models."""
    print("\n📊 Generating comprehensive comparison charts...")
    
    # Load all individual results
    all_results = []
    for result_file in REPORTS_DIR.glob("*_results.json"):
        if result_file.name != "all_models_results.csv":
            with open(result_file, 'r') as f:
                all_results.append(json.load(f))
    
    if not all_results:
        print("❌ No trained models found. Train models first.")
        return
    
    # Sort by RMSE
    results_df = pd.DataFrame(all_results).sort_values('rmse')
    results_df.to_csv(REPORTS_DIR / "all_models_results.csv", index=False)
    
    # Generate comparison plots (simplified - just the summary bar charts)
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    
    # Bar chart comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    models = results_df['model'].tolist()
    colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
    
    axes[0, 0].barh(models, results_df['rmse'], color=colors)
    axes[0, 0].set_xlabel('RMSE')
    axes[0, 0].set_title('RMSE by Model (lower is better)')
    
    axes[0, 1].barh(models, results_df['mae'], color=colors)
    axes[0, 1].set_xlabel('MAE')
    axes[0, 1].set_title('MAE by Model (lower is better)')
    
    axes[1, 0].barh(models, results_df['r2'], color=colors)
    axes[1, 0].set_xlabel('R² Score')
    axes[1, 0].set_title('R² by Model (higher is better)')
    
    axes[1, 1].barh(models, results_df['params']/1e6, color=colors)
    axes[1, 1].set_xlabel('Parameters (Millions)')
    axes[1, 1].set_title('Model Size')
    
    plt.tight_layout()
    plt.savefig(COMPARISON_DIR / 'model_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 COMPARISON RESULTS")
    print("=" * 70)
    print("\n" + results_df[['model', 'rmse', 'mae', 'r2', 'params', 'epochs']].to_string(index=False))
    
    if len(results_df) > 0:
        best = results_df.iloc[0]
        print(f"\n🏆 BEST MODEL: {best['model']} ({best['params']/1e6:.1f}M params)")
        print(f"   RMSE: ${best['rmse']:.2f}, R²: {best['r2']:.4f}")
    
    print(f"\n📁 Comparison charts: {COMPARISON_DIR}")


def main():
    parser = argparse.ArgumentParser(description='Train deep learning models for price prediction')
    parser.add_argument('--model', type=str, help='Train a specific model (LSTM, GRU, Transformer, TCN, WaveNet, NBEATS, PatchTST)')
    parser.add_argument('--compare', action='store_true', help='Generate comparison charts for all trained models')
    parser.add_argument('--all', action='store_true', help='Train all models sequentially')
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("🚀 AGRICAST PRODUCTION TRAINING PIPELINE - v2.0")
    print("=" * 70)
    
    # Create directories
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    if args.compare:
        generate_comparison()
        return
    
    # Load data
    splits, scaler, n_feat = load_data()
    seq_len = SEQUENCE_LENGTH
    dates = splits['dates']
    
    if args.model:
        # Train single model
        train_single_model(args.model, splits, scaler, seq_len, n_feat, dates)
    elif args.all:
        # Train all models
        all_models = ['LSTM', 'GRU', 'Transformer', 'TCN', 'WaveNet', 'NBEATS', 'PatchTST']  # 7 models
        for i, name in enumerate(all_models, 1):
            print(f"\n[{i}/{len(all_models)}] Training {name}...")
            train_single_model(name, splits, scaler, seq_len, n_feat, dates)
        generate_comparison()
    else:
        print("\nUsage:")
        print("  python train_all.py --model LSTM     # Train single model")
        print("  python train_all.py --all            # Train all models")
        print("  python train_all.py --compare        # Generate comparison charts")
        print("\nAvailable models: LSTM, GRU, Transformer, TCN, WaveNet, NBEATS, PatchTST")


if __name__ == "__main__":
    main()

