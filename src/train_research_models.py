"""
Research-Grade Model Training Script

Train all advanced models for research paper evaluation.
Includes comprehensive metrics and visualization.

Usage:
    python train_research_models.py
    python train_research_models.py --models TFT Informer
    python train_research_models.py --epochs 300
"""

import os
import sys
import argparse
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    DATA_PATH, MODEL_DIR, FIGURES_DIR, REPORTS_DIR, get_config
)
from data_loader import DataLoader
from feature_engineering import FeatureEngineer
from models.advanced_models import (
    build_temporal_fusion_transformer,
    build_nbeats,
    build_informer,
    build_wavenet,
    build_deepar,
    build_tcn,
    build_hybrid_attention_network,
    print_model_summary
)


def setup_gpu():
    """Configure GPU for optimal training."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            print(f"✓ GPU enabled: {gpus[0].name}")
            print("✓ Mixed precision (FP16) enabled")
        except Exception as e:
            print(f"⚠ GPU setup error: {e}")
    else:
        print("⚠ No GPU found, using CPU")


def get_callbacks(model_name, patience=30):
    """Get training callbacks."""
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=patience // 3,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(MODEL_DIR, f'{model_name}_best.keras'),
            monitor='val_loss',
            save_best_only=True,
            verbose=0
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=f'logs/{model_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        )
    ]


def prepare_data(config):
    """Load and prepare training data."""
    print("\n📊 Preparing data...")
    
    data_loader = DataLoader(
        data_path=DATA_PATH,
        sequence_length=config.data.sequence_length,
        forecast_horizon=config.data.forecast_horizon,
        train_ratio=config.data.train_ratio,
        val_ratio=config.data.val_ratio,
        test_ratio=config.data.test_ratio
    )
    
    # Load and preprocess
    raw_df = data_loader.load_data()
    processed_df = data_loader.preprocess_data(raw_df, commodity=config.data.commodity)
    daily_df = data_loader.aggregate_daily_prices(processed_df, price_col='modal_price')
    
    # Feature engineering
    feature_engineer = FeatureEngineer(
        target_col='modal_price',
        lag_periods=config.data.lag_periods,
        rolling_windows=config.data.rolling_windows
    )
    featured_df = feature_engineer.create_all_features(daily_df, include_external=True)
    
    # Select features
    selected_features = feature_engineer.select_top_features(
        featured_df, n_features=config.data.top_n_features
    )
    
    # Prepare data
    data_dict = data_loader.prepare_data_for_training(
        featured_df, feature_cols=selected_features, target_col='modal_price'
    )
    
    # Handle empty validation set
    if len(data_dict['X_val']) == 0:
        print("⚠ No validation data, using training data")
        data_dict['X_val'] = data_dict['X_train']
        data_dict['y_val'] = data_dict['y_train']
    
    return data_loader, data_dict


def build_research_models(seq_length, n_features):
    """Build all research-grade models."""
    print("\n🔬 Building research-grade models...")
    
    models = {
        'TFT': build_temporal_fusion_transformer(seq_length, n_features),
        'NBEATS': build_nbeats(seq_length, n_features),
        'Informer': build_informer(seq_length, n_features),
        'WaveNet': build_wavenet(seq_length, n_features),
        'DeepAR': build_deepar(seq_length, n_features),
        'TCN': build_tcn(seq_length, n_features),
        'HybridAttn': build_hybrid_attention_network(seq_length, n_features),
    }
    
    total_params = sum(m.count_params() for m in models.values())
    
    for name, model in models.items():
        print(f"   {name}: {model.count_params():,} parameters")
    
    print(f"\n   Total parameters: {total_params:,}")
    
    return models


def train_all_models(models, data_dict, epochs=200, batch_size=32):
    """Train all models."""
    histories = {}
    
    X_train = data_dict['X_train']
    y_train = data_dict['y_train']
    X_val = data_dict['X_val']
    y_val = data_dict['y_val']
    
    # Adjust batch size for small datasets
    batch_size = min(batch_size, len(X_train))
    
    for name, model in models.items():
        print(f"\n{'='*60}")
        print(f"  Training {name}")
        print(f"{'='*60}")
        
        callbacks = get_callbacks(name)
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        histories[name] = history
        
        # Save final model
        model.save(os.path.join(MODEL_DIR, f'{name.lower()}_final.keras'))
        print(f"   ✓ {name} saved")
    
    return histories


def evaluate_models(models, data_dict, data_loader):
    """Evaluate all models."""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    X_test = data_dict['X_test']
    y_test = data_dict['y_test']
    
    # If no test data, use validation
    if len(X_test) == 0:
        X_test = data_dict['X_val']
        y_test = data_dict['y_val']
    
    y_true = data_loader.inverse_transform_predictions(y_test)
    
    results = []
    
    for name, model in models.items():
        y_pred_scaled = model.predict(X_test, verbose=0).flatten()
        y_pred = data_loader.inverse_transform_predictions(y_pred_scaled)
        
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else 0.0
        
        results.append({
            'Model': name,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'R2': r2,
            'Parameters': model.count_params()
        })
        
        print(f"\n{name}:")
        print(f"   RMSE: {rmse:.2f}, MAE: {mae:.2f}, MAPE: {mape:.2f}%, R²: {r2:.4f}")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(REPORTS_DIR, 'research_model_results.csv'), index=False)
    
    return results_df


def plot_comparison(histories, results_df):
    """Plot model comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Training curves
    for name, history in histories.items():
        axes[0, 0].plot(history.history['loss'], label=name)
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    for name, history in histories.items():
        axes[0, 1].plot(history.history['val_loss'], label=name)
    axes[0, 1].set_title('Validation Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # RMSE comparison
    models = results_df['Model'].values
    rmse_values = results_df['RMSE'].values
    colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
    
    bars = axes[1, 0].bar(models, rmse_values, color=colors)
    axes[1, 0].set_title('RMSE Comparison')
    axes[1, 0].set_ylabel('RMSE')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Parameters vs Performance
    params = results_df['Parameters'].values / 1e6
    axes[1, 1].scatter(params, rmse_values, c=colors, s=100)
    for i, name in enumerate(models):
        axes[1, 1].annotate(name, (params[i], rmse_values[i]), fontsize=8)
    axes[1, 1].set_title('Parameters vs RMSE')
    axes[1, 1].set_xlabel('Parameters (Millions)')
    axes[1, 1].set_ylabel('RMSE')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'research_model_comparison.png'), dpi=150)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Train research-grade models')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--models', nargs='+', default=None,
                       help='Specific models to train (default: all)')
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print(" 🔬 RESEARCH-GRADE MODEL TRAINING")
    print("=" * 60)
    
    # Setup
    setup_gpu()
    config = get_config()
    
    # Print available models
    print_model_summary()
    
    # Prepare data
    data_loader, data_dict = prepare_data(config)
    
    seq_length = data_dict['X_train'].shape[1]
    n_features = data_dict['X_train'].shape[2]
    
    print(f"\n📊 Data shapes:")
    print(f"   Train: {data_dict['X_train'].shape}")
    print(f"   Val: {data_dict['X_val'].shape}")
    print(f"   Test: {data_dict['X_test'].shape}")
    
    # Build models
    all_models = build_research_models(seq_length, n_features)
    
    # Filter models if specified
    if args.models:
        models = {k: v for k, v in all_models.items() if k in args.models}
    else:
        models = all_models
    
    # Train
    histories = train_all_models(
        models, data_dict,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # Evaluate
    print("\n" + "=" * 60)
    print(" 📊 MODEL EVALUATION")
    print("=" * 60)
    
    results_df = evaluate_models(models, data_dict, data_loader)
    
    print("\n" + "=" * 60)
    print(" RESULTS SUMMARY")
    print("=" * 60)
    print(results_df.to_string(index=False))
    
    # Best model
    best_idx = results_df['RMSE'].idxmin()
    best_model = results_df.loc[best_idx, 'Model']
    print(f"\n🏆 Best Model: {best_model}")
    
    # Plot comparison
    plot_comparison(histories, results_df)
    
    print("\n" + "=" * 60)
    print(" ✓ TRAINING COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
