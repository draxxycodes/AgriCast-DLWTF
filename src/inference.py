"""
Inference Script - Test Trained Models

Load trained models and make predictions on sample data.

Usage:
    python inference.py
    python inference.py --model lstm
    python inference.py --model gru
    python inference.py --model transformer
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import MODEL_DIR, DATA_PATH, get_config
from data_loader import DataLoader
from feature_engineering import FeatureEngineer

# Import model builders to reconstruct models
from models.lstm_model import LSTMAttentionModel
from models.gru_model import BidirectionalGRUModel
from models.transformer_model import TransformerModel


def load_model(model_name: str, sequence_length: int, n_features: int, config):
    """Load a trained model by rebuilding architecture and loading weights."""
    model_path = os.path.join(MODEL_DIR, f'{model_name.lower()}_final.keras')
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return None
    
    print(f"✓ Loading {model_name} from {model_path}")
    
    # Try to load directly first
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        print(f"   ⚠ Direct load failed, rebuilding model architecture...")
    
    # Rebuild model and load weights
    if model_name.lower() == 'lstm':
        builder = LSTMAttentionModel(
            sequence_length=sequence_length,
            n_features=n_features,
            lstm_units=config.lstm.lstm_units,
            attention_units=config.lstm.attention_units,
            num_attention_heads=config.lstm.num_attention_heads,
            dense_units=config.lstm.dense_units,
            dropout_rate=config.lstm.dropout_rate,
            l2_reg=config.lstm.l2_reg,
            learning_rate=config.lstm.learning_rate
        )
        model = builder.build_model()
    elif model_name.lower() == 'gru':
        builder = BidirectionalGRUModel(
            sequence_length=sequence_length,
            n_features=n_features,
            gru_units=config.gru.gru_units,
            conv_filters=config.gru.conv_filters,
            dense_units=config.gru.dense_units,
            dropout_rate=config.gru.dropout_rate,
            l2_reg=config.gru.l2_reg,
            learning_rate=config.gru.learning_rate
        )
        model = builder.build_model()
    elif model_name.lower() == 'transformer':
        builder = TransformerModel(
            sequence_length=sequence_length,
            n_features=n_features,
            d_model=config.transformer.d_model,
            num_heads=config.transformer.num_heads,
            num_layers=config.transformer.num_layers,
            ff_dim=config.transformer.ff_dim,
            dense_units=config.transformer.dense_units,
            dropout_rate=config.transformer.dropout_rate,
            l2_reg=config.transformer.l2_reg,
            learning_rate=config.transformer.learning_rate
        )
        model = builder.build_model()
    else:
        print(f"❌ Unknown model: {model_name}")
        return None
    
    # Load weights
    try:
        model.load_weights(model_path)
        print(f"   ✓ Weights loaded successfully")
    except Exception as e:
        print(f"   ⚠ Could not load weights: {e}")
    
    return model


def prepare_test_data(config):
    """Prepare test data from the dataset."""
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
    processed_df = data_loader.preprocess_data(raw_df)
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
        featured_df,
        n_features=config.data.top_n_features
    )
    
    # Prepare data
    data_dict = data_loader.prepare_data_for_training(
        featured_df,
        feature_cols=selected_features,
        target_col='modal_price'
    )
    
    return data_loader, data_dict


def make_predictions(model, X_test, data_loader):
    """Make predictions and transform to original scale."""
    predictions_scaled = model.predict(X_test, verbose=0).flatten()
    predictions = data_loader.inverse_transform_predictions(predictions_scaled)
    return predictions


def evaluate_predictions(y_true, y_pred):
    """Calculate evaluation metrics."""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else 0.0
    
    return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R2': r2}


def main():
    parser = argparse.ArgumentParser(description='Test trained models')
    parser.add_argument('--model', type=str, default='all',
                        choices=['lstm', 'gru', 'transformer', 'all'],
                        help='Model to test')
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print(" 🧪 MODEL INFERENCE / TESTING")
    print("=" * 60)
    
    # Load config
    config = get_config()
    
    # Prepare data
    print("\nPreparing test data...")
    data_loader, data_dict = prepare_test_data(config)
    
    X_test = data_dict['X_test']
    y_test = data_dict['y_test']
    
    # Inverse transform ground truth
    y_test_original = data_loader.inverse_transform_predictions(y_test)
    
    sequence_length = data_dict['X_train'].shape[1]
    n_features = data_dict['X_train'].shape[2]
    
    print(f"\n✓ Test samples: {len(X_test)}")
    print(f"✓ Sequence length: {sequence_length}, Features: {n_features}")
    
    # Models to test
    if args.model == 'all':
        models_to_test = ['lstm', 'gru', 'transformer']
    else:
        models_to_test = [args.model]
    
    results = {}
    
    for model_name in models_to_test:
        print(f"\n{'=' * 40}")
        print(f" Testing {model_name.upper()}")
        print(f"{'=' * 40}")
        
        model = load_model(model_name, sequence_length, n_features, config)
        if model is None:
            continue
        
        # Make predictions
        predictions = make_predictions(model, X_test, data_loader)
        
        # Evaluate
        metrics = evaluate_predictions(y_test_original, predictions)
        results[model_name] = metrics
        
        print(f"\n📊 Results for {model_name.upper()}:")
        print(f"   RMSE:  {metrics['RMSE']:.2f}")
        print(f"   MAE:   {metrics['MAE']:.2f}")
        print(f"   MAPE:  {metrics['MAPE']:.2f}%")
        print(f"   R²:    {metrics['R2']:.4f}")
        
        print(f"\n📈 Predictions vs Actual:")
        for i in range(min(5, len(predictions))):
            print(f"   Sample {i+1}: Predicted = {predictions[i]:.2f}, Actual = {y_test_original[i]:.2f}")
    
    # Summary
    if len(results) > 1:
        print("\n" + "=" * 60)
        print(" COMPARISON SUMMARY")
        print("=" * 60)
        
        summary_df = pd.DataFrame(results).T
        print(summary_df.to_string())
        
        best_model = min(results.keys(), key=lambda k: results[k]['RMSE'])
        print(f"\n🏆 Best Model (lowest RMSE): {best_model.upper()}")
    
    print("\n" + "=" * 60)
    print(" ✓ TESTING COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
