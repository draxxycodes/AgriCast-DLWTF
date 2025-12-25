"""
Hybrid Ensemble Model - Stacking Meta-Learner

Combines the 4 best performing trained models:
- GRU (R²=0.41, 30.11M params) 
- WaveNet (R²=0.30, 5.83M params)
- TFT (R²=0.25, 2.13M params)
- LSTM (R²=0.22, 11.35M params)

Total combined params: ~50M
Architecture: Stacking with trainable meta-learner
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

# TensorFlow setup
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"✓ GPU: {gpus[0].name}")

import keras
from keras import layers, Model
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# =====================================================
# CONFIG
# =====================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "daily_prices.csv"
MODEL_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures" / "hybrid"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"

SEQUENCE_LENGTH = 60
META_EPOCHS = 100
BATCH_SIZE = 32
PATIENCE = 25
LEARNING_RATE = 5e-5

# Base models to ensemble (ranked by R²)
BASE_MODELS = [
    ('GRU', 'gru.keras', 0.41),
    ('WaveNet', 'wavenet.keras', 0.30),
    ('TFT', 'tft.keras', 0.25),
    ('LSTM', 'lstm.keras', 0.22),
]

# =====================================================
# DATA LOADING (same as train_all.py)
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
    
    print(f"   Train: {len(splits['X_train'])}, Val: {len(splits['X_val'])}, Test: {len(splits['X_test'])}")
    print(f"   Features: {X.shape[2]}, Sequence: {SEQUENCE_LENGTH}")
    
    return splits, scaler


# =====================================================
# LOAD PRE-TRAINED BASE MODELS
# =====================================================
def load_base_models():
    """Load all pre-trained base models."""
    print("\n🔧 Loading pre-trained base models...")
    
    models = {}
    total_params = 0
    
    for name, filename, r2 in BASE_MODELS:
        path = MODEL_DIR / filename
        if path.exists():
            try:
                model = keras.models.load_model(path, compile=False)
                models[name] = model
                params = model.count_params()
                total_params += params
                print(f"   ✓ {name}: {params:,} params (R²={r2:.2f})")
            except Exception as e:
                print(f"   ✗ {name}: Failed to load - {e}")
        else:
            print(f"   ✗ {name}: File not found at {path}")
    
    print(f"\n   Total base model params: {total_params:,} ({total_params/1e6:.1f}M)")
    return models


# =====================================================
# GET BASE MODEL PREDICTIONS
# =====================================================
def get_base_predictions(models, X):
    """Get predictions from all base models."""
    predictions = {}
    for name, model in models.items():
        try:
            pred = model.predict(X, verbose=0, batch_size=64).flatten()
            # Handle NaN predictions
            if np.isnan(pred).any():
                print(f"   ⚠ {name}: Contains NaN, replacing with mean")
                pred = np.nan_to_num(pred, nan=np.nanmean(pred))
            predictions[name] = pred
        except Exception as e:
            print(f"   ✗ {name}: Prediction failed - {e}")
    return predictions


# =====================================================
# BUILD META-LEARNER
# =====================================================
def build_meta_learner(n_base_models):
    """Build trainable meta-learner for stacking."""
    inputs = layers.Input(shape=(n_base_models,), name='base_predictions')
    
    # Feature expansion
    x = layers.Dense(64, activation='gelu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Dense(128, activation='gelu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Dense(64, activation='gelu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    
    x = layers.Dense(32, activation='gelu')(x)
    
    # Residual from weighted average of inputs
    weights = layers.Dense(n_base_models, activation='softmax', name='learned_weights')(inputs)
    weighted_avg = layers.Dot(axes=1)([inputs, weights])
    
    # Combine meta-learner output with weighted average
    combined = layers.Concatenate()([x, weighted_avg])
    x = layers.Dense(16, activation='gelu')(combined)
    
    outputs = layers.Dense(1, dtype='float32', name='final_output')(x)
    
    model = Model(inputs, outputs, name='MetaLearner')
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=LEARNING_RATE),
        loss='huber',
        metrics=['mae']
    )
    
    return model


# =====================================================
# BUILD END-TO-END HYBRID MODEL
# =====================================================
def build_hybrid_model(base_models, seq_len, n_feat):
    """Build end-to-end hybrid model with frozen base models and trainable meta-learner."""
    
    # Input layer
    inputs = layers.Input(shape=(seq_len, n_feat), name='input')
    
    # Get predictions from each base model (frozen)
    base_outputs = []
    for name, model in base_models.items():
        # Freeze base model weights
        model.trainable = False
        
        # Get output
        output = model(inputs)
        if len(output.shape) > 1:
            output = layers.Flatten()(output)
        base_outputs.append(output)
    
    # Stack base model outputs
    if len(base_outputs) > 1:
        stacked = layers.Concatenate(name='stacked_predictions')(base_outputs)
    else:
        stacked = base_outputs[0]
    
    # Meta-learner on top
    x = layers.Dense(128, activation='gelu')(stacked)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(64, activation='gelu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Dense(32, activation='gelu')(x)
    x = layers.Dropout(0.1)(x)
    
    # Learned weights for ensemble
    weights = layers.Dense(len(base_outputs), activation='softmax', name='ensemble_weights')(stacked)
    weighted = layers.Dot(axes=1)([stacked, weights])
    
    # Final combination
    combined = layers.Concatenate()([x, weighted])
    x = layers.Dense(16, activation='gelu')(combined)
    outputs = layers.Dense(1, dtype='float32', name='output')(x)
    
    model = Model(inputs, outputs, name='HybridEnsemble')
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=LEARNING_RATE),
        loss='huber',
        metrics=['mae']
    )
    
    return model


# =====================================================
# EVALUATION
# =====================================================
def evaluate(predictions, y_true, scaler):
    """Evaluate predictions."""
    # Inverse transform
    dummy = np.zeros((len(predictions), scaler.n_features_in_))
    dummy[:, 0] = predictions
    y_pred_orig = scaler.inverse_transform(dummy)[:, 0]
    
    dummy[:, 0] = y_true
    y_true_orig = scaler.inverse_transform(dummy)[:, 0]
    
    rmse = np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))
    mae = mean_absolute_error(y_true_orig, y_pred_orig)
    r2 = r2_score(y_true_orig, y_pred_orig)
    mape = np.mean(np.abs((y_true_orig - y_pred_orig) / (y_true_orig + 1e-8))) * 100
    
    return {
        'rmse': rmse, 'mae': mae, 'r2': r2, 'mape': mape,
        'y_pred': y_pred_orig, 'y_true': y_true_orig
    }


# =====================================================
# PLOT RESULTS
# =====================================================
def plot_results(metrics, base_metrics, dates, figures_dir):
    """Plot hybrid model results and comparison."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Predictions vs Actual
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(dates[-len(metrics['y_true']):], metrics['y_true'], 'b-', 
            label='Actual', linewidth=2, alpha=0.9)
    ax.plot(dates[-len(metrics['y_pred']):], metrics['y_pred'], 'r--', 
            label=f'Hybrid (R²={metrics["r2"]:.3f})', linewidth=2, alpha=0.8)
    ax.set_xlabel('Date')
    ax.set_ylabel('Price ($)')
    ax.set_title('Hybrid Ensemble Model - Predictions vs Actual', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / 'predictions.png', dpi=150)
    plt.close()
    
    # 2. Comparison with base models
    models = list(base_metrics.keys()) + ['Hybrid']
    r2_scores = [base_metrics[m]['r2'] for m in base_metrics] + [metrics['r2']]
    rmse_scores = [base_metrics[m]['rmse'] for m in base_metrics] + [metrics['rmse']]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = ['#3498db'] * len(base_metrics) + ['#2ecc71']
    
    axes[0].barh(models, r2_scores, color=colors, edgecolor='black', alpha=0.8)
    axes[0].set_xlabel('R² Score')
    axes[0].set_title('R² Comparison (Higher = Better)', fontweight='bold')
    axes[0].axvline(0, color='red', linestyle='--', alpha=0.5)
    for i, v in enumerate(r2_scores):
        axes[0].text(v + 0.01, i, f'{v:.3f}', va='center')
    
    axes[1].barh(models, rmse_scores, color=colors, edgecolor='black', alpha=0.8)
    axes[1].set_xlabel('RMSE ($)')
    axes[1].set_title('RMSE Comparison (Lower = Better)', fontweight='bold')
    for i, v in enumerate(rmse_scores):
        axes[1].text(v + 5, i, f'${v:.0f}', va='center')
    
    plt.suptitle('Hybrid Model vs Base Models', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(figures_dir / 'comparison.png', dpi=150)
    plt.close()
    
    # 3. Error distribution
    errors = metrics['y_pred'] - metrics['y_true']
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(errors, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax.axvline(np.mean(errors), color='green', linestyle='-', linewidth=2, 
               label=f'Mean Error: ${np.mean(errors):.1f}')
    ax.set_xlabel('Prediction Error ($)')
    ax.set_ylabel('Frequency')
    ax.set_title('Hybrid Model - Error Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / 'error_distribution.png', dpi=150)
    plt.close()
    
    print(f"   ✓ Figures saved to: {figures_dir}")


# =====================================================
# MAIN
# =====================================================
def main():
    print("=" * 60)
    print("🚀 HYBRID ENSEMBLE MODEL - STACKING META-LEARNER")
    print("=" * 60)
    
    # Load data
    splits, scaler = load_data()
    
    # Load pre-trained base models
    base_models = load_base_models()
    
    if len(base_models) < 2:
        print("\n❌ Error: Need at least 2 base models. Exiting.")
        return
    
    # Get base model predictions on all splits
    print("\n📈 Getting base model predictions...")
    
    train_preds = get_base_predictions(base_models, splits['X_train'])
    val_preds = get_base_predictions(base_models, splits['X_val'])
    test_preds = get_base_predictions(base_models, splits['X_test'])
    
    # Stack predictions as features for meta-learner
    model_names = list(train_preds.keys())
    X_train_meta = np.column_stack([train_preds[m] for m in model_names])
    X_val_meta = np.column_stack([val_preds[m] for m in model_names])
    X_test_meta = np.column_stack([test_preds[m] for m in model_names])
    
    print(f"   Meta-learner input shape: {X_train_meta.shape}")
    
    # Evaluate individual base models on test set
    print("\n📊 Evaluating base models...")
    base_metrics = {}
    for name in model_names:
        pred = test_preds[name]
        metrics = evaluate(pred, splits['y_test'], scaler)
        base_metrics[name] = metrics
        print(f"   {name}: R²={metrics['r2']:.4f}, RMSE=${metrics['rmse']:.2f}")
    
    # Build and train meta-learner
    print("\n🔧 Training meta-learner...")
    meta_model = build_meta_learner(len(model_names))
    
    total_params = meta_model.count_params()
    for model in base_models.values():
        total_params += model.count_params()
    print(f"   Meta-learner params: {meta_model.count_params():,}")
    print(f"   Total hybrid params: {total_params:,} ({total_params/1e6:.1f}M)")
    
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=PATIENCE, 
            restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=10, 
            min_lr=1e-7, verbose=1
        )
    ]
    
    history = meta_model.fit(
        X_train_meta, splits['y_train'],
        validation_data=(X_val_meta, splits['y_val']),
        epochs=META_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate hybrid model
    print("\n📊 Evaluating hybrid model...")
    hybrid_pred = meta_model.predict(X_test_meta, verbose=0).flatten()
    hybrid_metrics = evaluate(hybrid_pred, splits['y_test'], scaler)
    
    print(f"\n{'='*60}")
    print("📊 FINAL RESULTS")
    print(f"{'='*60}")
    print(f"\n   HYBRID ENSEMBLE:")
    print(f"   ✓ R² Score:  {hybrid_metrics['r2']:.4f}")
    print(f"   ✓ RMSE:      ${hybrid_metrics['rmse']:.2f}")
    print(f"   ✓ MAE:       ${hybrid_metrics['mae']:.2f}")
    print(f"   ✓ MAPE:      {hybrid_metrics['mape']:.2f}%")
    print(f"   ✓ Params:    {total_params:,}")
    
    # Compare with best base model
    best_base = max(base_metrics.items(), key=lambda x: x[1]['r2'])
    improvement = hybrid_metrics['r2'] - best_base[1]['r2']
    print(f"\n   vs Best Base ({best_base[0]}):")
    print(f"   {'↑' if improvement > 0 else '↓'} R² change: {improvement:+.4f}")
    
    # Save model
    hybrid_path = MODEL_DIR / "hybrid_ensemble.keras"
    meta_model.save(hybrid_path)
    print(f"\n   ✓ Model saved: {hybrid_path}")
    
    # Plot results
    test_dates = splits['dates'][-len(splits['X_test']):]
    plot_results(hybrid_metrics, base_metrics, test_dates, FIGURES_DIR)
    
    # Update results CSV
    results_df = pd.read_csv(REPORTS_DIR / "all_models_results.csv")
    new_row = pd.DataFrame([{
        'model': 'Hybrid',
        'rmse': hybrid_metrics['rmse'],
        'mae': hybrid_metrics['mae'],
        'mape': hybrid_metrics['mape'],
        'r2': hybrid_metrics['r2'],
        'params': total_params,
        'epochs': len(history.history['loss'])
    }])
    results_df = pd.concat([results_df, new_row], ignore_index=True)
    results_df = results_df.sort_values('r2', ascending=False)
    results_df.to_csv(REPORTS_DIR / "all_models_results.csv", index=False)
    print(f"   ✓ Results updated: {REPORTS_DIR / 'all_models_results.csv'}")
    
    print(f"\n{'='*60}")
    if hybrid_metrics['r2'] > best_base[1]['r2']:
        print(f"🏆 HYBRID MODEL IS THE NEW BEST! (R²={hybrid_metrics['r2']:.4f})")
    else:
        print(f"✓ Training complete. Best model remains: {best_base[0]}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
