"""
Hybrid Ensemble Model - Stacking Meta-Learner

Dynamically combines the top 3 performing trained models
based on validation R² scores.

Available models: LSTM, GRU, Transformer, WaveNet, NBEATS
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
DATA_PATH = PROJECT_ROOT / "data" / "processed_agricultural.csv"  # Use preprocessed data
MODEL_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures" / "hybrid"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"

SEQUENCE_LENGTH = 60
META_EPOCHS = 100
BATCH_SIZE = 32
PATIENCE = 25
LEARNING_RATE = 5e-5

# Available models - will dynamically select top 3 based on results
AVAILABLE_MODELS = ['LSTM', 'GRU', 'Transformer', 'TCN', 'WaveNet', 'NBEATS', 'PatchTST']
TOP_N_MODELS = 3  # Number of models to combine


# =====================================================
# DATA LOADING (same format as train_all.py)
# =====================================================
def load_data():
    """Load and prepare the preprocessed agricultural dataset."""
    print("\n📊 Loading preprocessed agricultural data...")
    
    df = pd.read_csv(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date', 'price_normalized'])
    df = df.sort_values('date').reset_index(drop=True)
    
    print(f"   Records: {len(df):,}")
    print(f"   Commodities: {df['commodity_clean'].nunique()}")
    
    # Select feature columns - EXCLUDE the target to prevent leakage
    exclude_cols = ['date', 'price', 'price_log', 'price_normalized', 'commodity_clean', 
                    'commodity_mean', 'commodity_std', 'log_return']
    feature_cols = [col for col in df.columns if col not in exclude_cols 
                    and df[col].dtype in ['float64', 'float32', 'int64', 'int32']]
    
    # Prepare features
    features = df[feature_cols].copy()
    features = features.replace([np.inf, -np.inf], 0).fillna(0)
    
    scaler = RobustScaler()
    scaled = scaler.fit_transform(features.values).astype(np.float32)
    
    # Target: price_normalized (NOT log_return - those are nearly random!)
    y = df['price_normalized'].values.astype(np.float32)
    
    # Create sequences
    X = []
    y_seq = []
    for i in range(len(scaled) - SEQUENCE_LENGTH):
        X.append(scaled[i:i+SEQUENCE_LENGTH])
        y_seq.append(y[i+SEQUENCE_LENGTH])
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y_seq, dtype=np.float32)
    
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
    
    print(f"   Train: {len(splits['X_train']):,}, Val: {len(splits['X_val']):,}, Test: {len(splits['X_test']):,}")
    print(f"   Features: {X.shape[2]}, Sequence: {SEQUENCE_LENGTH}")
    
    return splits, scaler


# =====================================================
# LOAD PRE-TRAINED BASE MODELS (Dynamic selection)
# =====================================================
def load_base_models():
    """Load available pre-trained models and return top N by R² from results."""
    import json
    
    print(f"\n🔧 Loading pre-trained base models (will select top {TOP_N_MODELS})...")
    
    models = {}
    model_scores = {}
    
    # Try to load results to rank models
    for name in AVAILABLE_MODELS:
        filename = f"{name.lower()}.keras"
        path = MODEL_DIR / filename
        result_path = REPORTS_DIR / f"{name.lower()}_results.json"
        
        if path.exists():
            try:
                model = keras.models.load_model(path, compile=False)
                params = model.count_params()
                
                # Get R² score from results if available
                r2 = 0.0
                if result_path.exists():
                    with open(result_path, 'r') as f:
                        result = json.load(f)
                        r2 = result.get('r2', 0.0)
                
                models[name] = model
                model_scores[name] = {'r2': r2, 'params': params}
                print(f"   ✓ {name}: {params:,} params, R²={r2:.4f}")
            except Exception as e:
                print(f"   ✗ {name}: Failed to load - {e}")
        else:
            print(f"   ✗ {name}: Not trained yet ({path.name} not found)")
    
    if len(models) < TOP_N_MODELS:
        print(f"\n   ⚠ Only {len(models)} models available, need at least {TOP_N_MODELS}")
        return models, model_scores
    
    # Select top N by R² score
    sorted_models = sorted(model_scores.items(), key=lambda x: x[1]['r2'], reverse=True)
    top_models = sorted_models[:TOP_N_MODELS]
    
    print(f"\n   📊 Selected top {TOP_N_MODELS} models:")
    selected_models = {}
    total_params = 0
    for name, score in top_models:
        selected_models[name] = models[name]
        total_params += score['params']
        print(f"      • {name}: R²={score['r2']:.4f}, {score['params']/1e6:.1f}M params")
    
    print(f"   Total base params: {total_params:,} ({total_params/1e6:.1f}M)")
    return selected_models, model_scores


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
    
    # Load pre-trained base models (returns top 3 by R²)
    base_models, model_scores = load_base_models()
    
    if len(base_models) < 2:
        print("\n❌ Error: Need at least 2 trained base models. Train models first with:")
        print("   python train_all.py --all")
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
