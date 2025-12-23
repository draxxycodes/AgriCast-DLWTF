"""
Model Registry - All available forecasting models

Statistical Models:
- ARIMA (Auto-ARIMA / SARIMA)
- Prophet (Facebook Prophet)
- Exponential Smoothing (Holt-Winters)

Deep Learning Models:
- LSTM (with Attention)
- GRU (Bidirectional)
- Transformer
- TFT (Temporal Fusion Transformer)
- N-BEATS (Neural Basis Expansion)
- WaveNet (Dilated Convolutions)
- TCN (Temporal Convolutional Network)
"""

# Base classes
from .base_model import BaseModel, StatisticalModel, DeepLearningModel

# Statistical models
from .arima_model import ARIMAModel, create_arima_model
from .prophet_model import ProphetModel, create_prophet_model
from .exponential_smoothing import ExponentialSmoothingModel, create_exponential_smoothing_model

# Deep learning models
from .lstm_model import LSTMAttentionModel, create_lstm_attention_model
from .gru_model import BidirectionalGRUModel, create_bidirectional_gru_model
from .transformer_model import TransformerModel, create_transformer_model
from .temporal_fusion import TFTModel, create_tft_model
from .nbeats_model import NBEATSModel, create_nbeats_model
from .wavenet_model import WaveNetModel, create_wavenet_model
from .tcn_model import TCNModel, create_tcn_model

# Ensemble
from .ensemble_model import EnsembleModel

# Model registry
STATISTICAL_MODELS = {
    'arima': ARIMAModel,
    'prophet': ProphetModel,
    'exponential_smoothing': ExponentialSmoothingModel,
}

DEEP_LEARNING_MODELS = {
    'lstm': LSTMAttentionModel,
    'gru': BidirectionalGRUModel,
    'transformer': TransformerModel,
    'tft': TFTModel,
    'nbeats': NBEATSModel,
    'wavenet': WaveNetModel,
    'tcn': TCNModel,
}

ALL_MODELS = {**STATISTICAL_MODELS, **DEEP_LEARNING_MODELS}


def get_model(name: str, **kwargs):
    """Get a model by name."""
    name_lower = name.lower()
    if name_lower not in ALL_MODELS:
        raise ValueError(f"Unknown model: {name}. Available: {list(ALL_MODELS.keys())}")
    return ALL_MODELS[name_lower](**kwargs)


def list_models():
    """List all available models."""
    print("\n=== Statistical Models ===")
    for name in STATISTICAL_MODELS:
        print(f"  - {name}")
    
    print("\n=== Deep Learning Models ===")
    for name in DEEP_LEARNING_MODELS:
        print(f"  - {name}")


__all__ = [
    # Base
    'BaseModel', 'StatisticalModel', 'DeepLearningModel',
    
    # Statistical
    'ARIMAModel', 'ProphetModel', 'ExponentialSmoothingModel',
    
    # Deep Learning
    'LSTMAttentionModel', 'BidirectionalGRUModel', 'TransformerModel',
    'TFTModel', 'NBEATSModel', 'WaveNetModel', 'TCNModel',
    
    # Ensemble
    'EnsembleModel',
    
    # Registry
    'get_model', 'list_models', 'ALL_MODELS',
]
