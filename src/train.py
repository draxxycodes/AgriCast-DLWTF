"""
Main Training Script for Agricultural Price Prediction

Train individual models or compare all models.

Usage:
    python train.py --model lstm
    python train.py --model arima
    python train.py --all
    python train.py --compare
"""

import os
import sys
import argparse
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# TensorFlow setup
import tensorflow as tf
if tf.config.list_physical_devices('GPU'):
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.keras.mixed_precision.set_global_policy('mixed_float16')

from config import DATA_PATH, MODEL_DIR, FIGURES_DIR, REPORTS_DIR, get_config
from data_loader import DataLoader
from feature_engineering import FeatureEngineer
from models import list_models, get_model, DEEP_LEARNING_MODELS, STATISTICAL_MODELS


def prepare_data(config):
    """Load and prepare training data."""
    print("\n📊 Loading data...")
    
    data_loader = DataLoader(
        data_path=DATA_PATH,
        sequence_length=config.data.sequence_length,
        train_ratio=config.data.train_ratio,
        val_ratio=config.data.val_ratio,
        test_ratio=config.data.test_ratio
    )
    
    raw_df = data_loader.load_data()
    processed_df = data_loader.preprocess_data(raw_df)
    daily_df = data_loader.aggregate_daily_prices(processed_df, price_col='modal_price')
    
    # Feature engineering
    feature_engineer = FeatureEngineer(
        target_col='modal_price',
        lag_periods=config.data.lag_periods,
        rolling_windows=config.data.rolling_windows
    )
    featured_df = feature_engineer.create_all_features(daily_df, include_external=True)
    
    selected_features = feature_engineer.select_top_features(
        featured_df, n_features=config.data.top_n_features
    )
    
    data_dict = data_loader.prepare_data_for_training(
        featured_df, feature_cols=selected_features, target_col='modal_price'
    )
    
    # Handle empty validation
    if len(data_dict['X_val']) == 0:
        data_dict['X_val'] = data_dict['X_train']
        data_dict['y_val'] = data_dict['y_train']
    
    print(f"✓ Train: {data_dict['X_train'].shape}")
    print(f"✓ Val: {data_dict['X_val'].shape}")
    print(f"✓ Test: {data_dict['X_test'].shape}")
    
    return data_loader, data_dict, daily_df


def get_callbacks(model_name, patience=30):
    """Get training callbacks for deep learning models."""
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=patience // 3,
            min_lr=1e-7
        ),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(MODEL_DIR, f'{model_name}_best.keras'),
            monitor='val_loss',
            save_best_only=True
        )
    ]


def train_statistical_model(model_name, daily_df, config):
    """Train a statistical model."""
    print(f"\n{'='*50}")
    print(f"Training {model_name.upper()}")
    print(f"{'='*50}")
    
    y = daily_df['modal_price'].values
    
    if model_name == 'arima':
        from models import ARIMAModel
        model = ARIMAModel(seasonal=True, m=7)
        result = model.fit(y)
        print(f"✓ Order: {result['order']}, AIC: {result['aic']:.2f}")
        
    elif model_name == 'prophet':
        from models import ProphetModel
        model = ProphetModel(weekly_seasonality=True)
        result = model.fit(daily_df, date_col='date', target_col='modal_price')
        print(f"✓ Fitted on {result['n_samples']} samples")
        
    elif model_name == 'exponential_smoothing':
        from models import ExponentialSmoothingModel
        model = ExponentialSmoothingModel(trend='add', seasonal='add', seasonal_periods=7)
        result = model.fit(y)
        print(f"✓ Fitted")
    
    return model


def train_deep_learning_model(model_name, data_dict, config):
    """Train a deep learning model."""
    print(f"\n{'='*50}")
    print(f"Training {model_name.upper()}")
    print(f"{'='*50}")
    
    seq_length = data_dict['X_train'].shape[1]
    n_features = data_dict['X_train'].shape[2]
    
    # Create model
    if model_name == 'lstm':
        from models import LSTMAttentionModel
        model = LSTMAttentionModel(seq_length, n_features)
        model.build_model()
        model = model.model
    elif model_name == 'gru':
        from models import BidirectionalGRUModel
        model = BidirectionalGRUModel(seq_length, n_features)
        model.build_model()
        model = model.model
    elif model_name == 'transformer':
        from models import TransformerModel
        model = TransformerModel(seq_length, n_features)
        model.build_model()
        model = model.model
    elif model_name == 'tft':
        from models import create_tft_model
        wrapper = create_tft_model(seq_length, n_features)
        model = wrapper.model
    elif model_name == 'nbeats':
        from models import create_nbeats_model
        wrapper = create_nbeats_model(seq_length, n_features)
        model = wrapper.model
    elif model_name == 'wavenet':
        from models import create_wavenet_model
        wrapper = create_wavenet_model(seq_length, n_features)
        model = wrapper.model
    elif model_name == 'tcn':
        from models import create_tcn_model
        wrapper = create_tcn_model(seq_length, n_features)
        model = wrapper.model
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    print(f"✓ Parameters: {model.count_params():,}")
    
    # Train
    batch_size = min(32, len(data_dict['X_train']))
    
    history = model.fit(
        data_dict['X_train'], data_dict['y_train'],
        validation_data=(data_dict['X_val'], data_dict['y_val']),
        epochs=config.training.epochs,
        batch_size=batch_size,
        callbacks=get_callbacks(model_name),
        verbose=1
    )
    
    # Save
    model.save(os.path.join(MODEL_DIR, f'{model_name}_final.keras'))
    print(f"✓ Model saved")
    
    return model, history


def evaluate_model(model, X_test, y_test, data_loader, model_name):
    """Evaluate a deep learning model."""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    y_pred_scaled = model.predict(X_test, verbose=0).flatten()
    y_pred = data_loader.inverse_transform_predictions(y_pred_scaled)
    y_true = data_loader.inverse_transform_predictions(y_test)
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else 0.0
    
    print(f"\n{model_name} Results:")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE:  {mae:.2f}")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  R²:   {r2:.4f}")
    
    return {'model': model_name, 'rmse': rmse, 'mae': mae, 'mape': mape, 'r2': r2}


def main():
    parser = argparse.ArgumentParser(description='Train forecasting models')
    parser.add_argument('--model', type=str, default='lstm',
                       help='Model to train (lstm, gru, transformer, tft, nbeats, wavenet, tcn, arima, prophet)')
    parser.add_argument('--all-dl', action='store_true', help='Train all deep learning models')
    parser.add_argument('--all-stat', action='store_true', help='Train all statistical models')
    parser.add_argument('--compare', action='store_true', help='Compare all models')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--list', action='store_true', help='List available models')
    args = parser.parse_args()
    
    if args.list:
        list_models()
        return
    
    print("\n" + "=" * 60)
    print(" 🌾 AGRICULTURAL PRICE PREDICTION - MODEL TRAINING")
    print("=" * 60)
    
    config = get_config()
    config.training.epochs = args.epochs
    
    data_loader, data_dict, daily_df = prepare_data(config)
    
    results = []
    
    if args.model.lower() in STATISTICAL_MODELS:
        model = train_statistical_model(args.model.lower(), daily_df, config)
    elif args.model.lower() in DEEP_LEARNING_MODELS:
        model, history = train_deep_learning_model(args.model.lower(), data_dict, config)
        result = evaluate_model(model, data_dict['X_test'], data_dict['y_test'], data_loader, args.model)
        results.append(result)
    else:
        print(f"Unknown model: {args.model}")
        list_models()
        return
    
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(REPORTS_DIR, 'training_results.csv'), index=False)
        print(f"\n✓ Results saved to {REPORTS_DIR}/training_results.csv")
    
    print("\n" + "=" * 60)
    print(" ✓ TRAINING COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
