"""
Main Entry Point for Agricultural Commodity Price Prediction

Run this script to train and evaluate all deep learning models.
Optimized for NVIDIA RTX 4060 with CUDA support.

Usage:
    python main.py                    # Full training pipeline
    python main.py --mode eda         # Only EDA
    python main.py --mode train       # Only training
    python main.py --mode eval        # Only evaluation
    python main.py --commodity Onion  # Train for specific commodity
"""

import os
import sys
import argparse
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def setup_gpu():
    """Configure GPU/CUDA settings for RTX 4060."""
    import tensorflow as tf
    from config import get_config
    
    config = get_config()
    
    print("\n🖥️  Setting up GPU...")
    
    # Check for GPU
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus and config.gpu.use_gpu:
        try:
            for gpu in gpus:
                # Enable memory growth
                if config.gpu.memory_growth:
                    tf.config.experimental.set_memory_growth(gpu, True)
                
                # Set memory limit if specified
                if config.gpu.gpu_memory_limit:
                    tf.config.set_logical_device_configuration(
                        gpu,
                        [tf.config.LogicalDeviceConfiguration(
                            memory_limit=config.gpu.gpu_memory_limit
                        )]
                    )
            
            # Enable mixed precision for faster training
            if config.gpu.mixed_precision:
                tf.keras.mixed_precision.set_global_policy('mixed_float16')
                print("   ✓ Mixed precision (FP16) enabled")
            
            # Enable XLA JIT compilation
            if config.gpu.xla_acceleration:
                tf.config.optimizer.set_jit(True)
                print("   ✓ XLA JIT compilation enabled")
            
            print(f"   ✓ GPU detected: {gpus[0].name}")
            print(f"   ✓ CUDA support enabled")
            
        except RuntimeError as e:
            print(f"   ⚠ GPU setup error: {e}")
    else:
        print("   ⚠ No GPU detected, using CPU")
    
    # Set random seeds
    import numpy as np
    np.random.seed(config.seed)
    tf.random.set_seed(config.seed)
    
    return config


def run_eda(config):
    """Run Exploratory Data Analysis."""
    print("\n" + "=" * 60)
    print(" EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    
    from eda import run_full_eda
    run_full_eda(config)


def prepare_data(config):
    """Load, preprocess, and engineer features for the data."""
    print("\n" + "=" * 60)
    print(" DATA PREPARATION")
    print("=" * 60)
    
    from data_loader import DataLoader
    from feature_engineering import FeatureEngineer
    from config import DATA_PATH
    
    # Initialize data loader
    data_loader = DataLoader(
        data_path=DATA_PATH,
        sequence_length=config.data.sequence_length,
        forecast_horizon=config.data.forecast_horizon,
        train_ratio=config.data.train_ratio,
        val_ratio=config.data.val_ratio,
        test_ratio=config.data.test_ratio,
        scaler_type=config.data.scaler_type
    )
    
    # Load and preprocess
    commodity_str = config.data.commodity or "ALL commodities"
    print(f"\nLoading data for: {commodity_str}")
    raw_df = data_loader.load_data()
    processed_df = data_loader.preprocess_data(
        raw_df,
        commodity=config.data.commodity
    )
    
    # Aggregate to daily prices
    daily_df = data_loader.aggregate_daily_prices(
        processed_df,
        price_col=config.data.target_col.lower().replace(' ', '_'),
        agg_method='mean'
    )
    
    print(f"✓ Daily records: {len(daily_df)}")
    
    # Feature Engineering
    print("\nEngineering features...")
    feature_engineer = FeatureEngineer(
        target_col='modal_price',
        lag_periods=config.data.lag_periods,
        rolling_windows=config.data.rolling_windows
    )
    
    featured_df = feature_engineer.create_all_features(daily_df, include_external=True)
    
    # Select top features
    feature_names = feature_engineer.get_feature_names()
    selected_features = feature_engineer.select_top_features(
        featured_df,
        n_features=config.data.top_n_features
    )
    
    # Prepare training data
    print("\nPreparing training data...")
    data_dict = data_loader.prepare_data_for_training(
        featured_df,
        feature_cols=selected_features,
        target_col='modal_price'
    )
    
    return data_loader, data_dict, selected_features


def build_models(config, n_features, sequence_length):
    """Build all model architectures."""
    print("\n" + "=" * 60)
    print(" BUILDING MODELS")
    print("=" * 60)
    
    from models.lstm_model import LSTMAttentionModel
    from models.gru_model import BidirectionalGRUModel
    from models.transformer_model import TransformerModel
    
    models = {}
    
    # LSTM with Attention
    print("\n📦 Building LSTM with Attention...")
    lstm_builder = LSTMAttentionModel(
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
    models['LSTM'] = lstm_builder.build_model()
    print(f"   ✓ LSTM parameters: {models['LSTM'].count_params():,}")
    
    # GRU
    print("\n📦 Building Bidirectional GRU...")
    gru_builder = BidirectionalGRUModel(
        sequence_length=sequence_length,
        n_features=n_features,
        gru_units=config.gru.gru_units,
        conv_filters=config.gru.conv_filters,
        dense_units=config.gru.dense_units,
        dropout_rate=config.gru.dropout_rate,
        l2_reg=config.gru.l2_reg,
        learning_rate=config.gru.learning_rate
    )
    models['GRU'] = gru_builder.build_model()
    print(f"   ✓ GRU parameters: {models['GRU'].count_params():,}")
    
    # Transformer
    print("\n📦 Building Transformer...")
    transformer_builder = TransformerModel(
        sequence_length=sequence_length,
        n_features=n_features,
        d_model=config.transformer.d_model,
        num_heads=config.transformer.num_heads,
        num_layers=config.transformer.num_layers,
        ff_dim=config.transformer.ff_dim,
        dense_units=config.transformer.dense_units,
        dropout_rate=config.transformer.dropout_rate,
        l2_reg=config.transformer.l2_reg,
        learning_rate=config.transformer.learning_rate,
        use_learnable_pos=config.transformer.use_learnable_pos
    )
    models['Transformer'] = transformer_builder.build_model()
    print(f"   ✓ Transformer parameters: {models['Transformer'].count_params():,}")
    
    return models


def train_models(config, models, data_dict):
    """Train all models."""
    print("\n" + "=" * 60)
    print(" TRAINING MODELS")
    print("=" * 60)
    
    from training import ModelTrainer, get_callbacks
    from config import MODEL_DIR
    
    X_train = data_dict['X_train']
    y_train = data_dict['y_train']
    X_val = data_dict['X_val']
    y_val = data_dict['y_val']
    
    trainer = ModelTrainer(
        checkpoint_dir=MODEL_DIR,
        log_dir=config.training.log_dir,
        epochs=config.training.epochs,
        batch_size=config.training.batch_size,
        patience=config.training.patience,
        verbose=config.training.verbose
    )
    
    histories = {}
    
    for name, model in models.items():
        print(f"\n🚀 Training {name}...")
        callbacks = get_callbacks(
            model_name=name.lower(),
            checkpoint_dir=MODEL_DIR,
            patience=config.training.patience,
            reduce_lr_patience=config.training.reduce_lr_patience,
            min_lr=config.training.min_lr
        )
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=config.training.epochs,
            batch_size=config.training.batch_size,
            callbacks=callbacks,
            verbose=config.training.verbose
        )
        
        histories[name] = history
        
        # Save model
        model.save(os.path.join(MODEL_DIR, f'{name.lower()}_final.keras'))
        print(f"   ✓ {name} saved to {MODEL_DIR}")
    
    return histories


def train_ensemble(config, models, data_dict):
    """Train ensemble model."""
    print("\n" + "=" * 60)
    print(" TRAINING ENSEMBLE")
    print("=" * 60)
    
    from models.ensemble_model import EnsembleModel
    
    X_train = data_dict['X_train']
    y_train = data_dict['y_train']
    X_val = data_dict['X_val']
    y_val = data_dict['y_val']
    
    # Handle empty validation set
    if len(X_val) == 0:
        print("⚠ No validation data available, using training data for ensemble weights")
        X_val = X_train
        y_val = y_train
    
    sequence_length = X_train.shape[1]
    n_features = X_train.shape[2]
    
    ensemble = EnsembleModel(
        sequence_length=sequence_length,
        n_features=n_features,
        ensemble_method=config.ensemble.ensemble_method
    )
    
    # Use trained models
    ensemble.base_models = models
    
    # Calculate weights
    print("\nCalculating validation-based weights...")
    ensemble.calculate_validation_weights(X_val, y_val)
    
    # Train meta-learner
    print("\nTraining meta-learner...")
    ensemble_history = ensemble.train_meta_learner(
        X_train, y_train,
        X_val, y_val,
        epochs=50,
        batch_size=min(config.training.batch_size, len(X_train))  # Ensure batch_size <= samples
    )
    
    return ensemble, ensemble_history


def evaluate_models(config, models, ensemble, data_dict, data_loader):
    """Evaluate all models and generate reports."""
    print("\n" + "=" * 60)
    print(" MODEL EVALUATION")
    print("=" * 60)
    
    from evaluation import ModelEvaluator, evaluate_all_models
    from config import FIGURES_DIR
    
    X_test = data_dict['X_test']
    y_test = data_dict['y_test']
    
    # Generate predictions
    print("\nGenerating predictions...")
    predictions = {}
    
    for name, model in models.items():
        pred = model.predict(X_test, verbose=0).flatten()
        predictions[name] = pred
    
    # Ensemble predictions
    predictions['Ensemble'] = ensemble.predict(X_test, method='stacking')
    predictions['Ensemble_Weighted'] = ensemble.predict(X_test, method='weighted')
    
    # Inverse transform to original scale
    y_test_original = data_loader.inverse_transform_predictions(y_test)
    predictions_original = {
        name: data_loader.inverse_transform_predictions(pred)
        for name, pred in predictions.items()
    }
    
    # Evaluate
    evaluator = ModelEvaluator(output_dir=FIGURES_DIR)
    metrics_df, summary = evaluate_all_models(
        y_test_original,
        predictions_original,
        output_dir=FIGURES_DIR
    )
    
    print(summary)
    
    return metrics_df, predictions_original


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Agricultural Commodity Price Prediction with Deep Learning'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='full',
        choices=['full', 'eda', 'train', 'eval'],
        help='Execution mode'
    )
    parser.add_argument(
        '--commodity',
        type=str,
        default=None,
        help='Commodity to predict (e.g., Tomato, Onion, Potato)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print(" 🌾 AGRICULTURAL COMMODITY PRICE PREDICTION")
    print(" Deep Learning with TensorFlow (CSE 3793)")
    print("=" * 60)
    
    # Setup GPU
    config = setup_gpu()
    
    # Override commodity if specified
    if args.commodity:
        config.data.commodity = args.commodity
    
    # Print configuration
    from config import print_config
    print_config(config)
    
    if args.mode in ['full', 'eda']:
        run_eda(config)
    
    if args.mode in ['full', 'train']:
        # Prepare data
        data_loader, data_dict, features = prepare_data(config)
        
        n_features = data_dict['X_train'].shape[2]
        sequence_length = data_dict['X_train'].shape[1]
        
        # Build models
        models = build_models(config, n_features, sequence_length)
        
        # Train models
        histories = train_models(config, models, data_dict)
        
        # Train ensemble
        ensemble, ensemble_history = train_ensemble(config, models, data_dict)
        
        if args.mode == 'full':
            # Evaluate
            metrics_df, predictions = evaluate_models(
                config, models, ensemble, data_dict, data_loader
            )
    
    elif args.mode == 'eval':
        print("\n⚠ Evaluation mode requires trained models. Run with --mode full first.")
    
    print("\n" + "=" * 60)
    print(" ✓ PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
