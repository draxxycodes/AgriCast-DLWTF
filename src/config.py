"""
Configuration for Agricultural Commodity Price Prediction

GPU: NVIDIA RTX 4060 with CUDA support
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple

# ==============================
# PATHS
# ==============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "daily_prices.csv")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data/processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")

# Create directories
for d in [PROCESSED_DATA_DIR, MODEL_DIR, FIGURES_DIR, REPORTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ==============================
# GPU CONFIGURATION (RTX 4060)
# ==============================
@dataclass
class GPUConfig:
    """GPU configuration for RTX 4060 CUDA training."""
    use_gpu: bool = True
    mixed_precision: bool = True  # Use FP16 for faster training
    memory_growth: bool = True     # Grow memory as needed
    gpu_memory_limit: int = None   # None = use all available, or set MB
    xla_acceleration: bool = True  # XLA JIT compilation


# ==============================
# DATA CONFIGURATION
# ==============================
@dataclass
class DataConfig:
    """Data loading and preprocessing configuration."""
    commodity: str = None          # None = use all commodities (more data)
    target_col: str = "Modal Price"
    sequence_length: int = 5       # Reduced for limited dataset (7 days only)
    forecast_horizon: int = 1      # Days to predict ahead
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    scaler_type: str = "minmax"    # 'minmax' or 'standard'
    
    # Feature engineering
    lag_periods: List[int] = field(default_factory=lambda: [1, 2, 3])
    rolling_windows: List[int] = field(default_factory=lambda: [3, 5])
    top_n_features: int = 20


# ==============================
# MODEL CONFIGURATIONS (IMPROVED TUNING)
# ==============================
@dataclass
class LSTMConfig:
    """LSTM with Attention model configuration."""
    lstm_units: Tuple[int, ...] = (128, 64, 32)
    attention_units: int = 64
    num_attention_heads: int = 4
    dense_units: Tuple[int, ...] = (64, 32)
    dropout_rate: float = 0.2          # Reduced from 0.3
    l2_reg: float = 0.0005             # Reduced regularization
    learning_rate: float = 0.0005      # Reduced from 0.001


@dataclass
class GRUConfig:
    """Bidirectional GRU model configuration."""
    gru_units: Tuple[int, ...] = (128, 64, 32)
    conv_filters: int = 64             # Increased from 32
    dense_units: Tuple[int, ...] = (128, 64)
    dropout_rate: float = 0.2          # Reduced from 0.3
    l2_reg: float = 0.0005             # Reduced regularization
    learning_rate: float = 0.0005      # Reduced from 0.001


@dataclass
class TransformerConfig:
    """Transformer model configuration."""
    d_model: int = 64
    num_heads: int = 4                 # Reduced for small dataset
    num_layers: int = 2                # Reduced for small dataset
    ff_dim: int = 128                  # Reduced for small dataset
    dense_units: Tuple[int, ...] = (64, 32)
    dropout_rate: float = 0.1
    l2_reg: float = 0.0005
    learning_rate: float = 0.0001
    use_learnable_pos: bool = True


@dataclass
class EnsembleConfig:
    """Ensemble model configuration."""
    include_lstm: bool = True
    include_gru: bool = True
    include_transformer: bool = True
    ensemble_method: str = "stacking"  # 'stacking', 'weighted', 'average'
    meta_hidden_units: Tuple[int, ...] = (32, 16)
    dropout_rate: float = 0.1          # Reduced from 0.2
    learning_rate: float = 0.0005


# ==============================
# TRAINING CONFIGURATION (IMPROVED)
# ==============================
@dataclass
class TrainingConfig:
    """Training configuration optimized for RTX 4060."""
    epochs: int = 200                  # Increased from 100
    batch_size: int = 32               # Reduced for small dataset
    patience: int = 30                 # Increased from 15
    reduce_lr_patience: int = 10       # Increased from 5
    min_lr: float = 1e-7
    warmup_epochs: int = 10            # Increased from 5
    
    # Callbacks
    use_tensorboard: bool = True
    use_early_stopping: bool = True
    use_model_checkpoint: bool = True
    use_lr_scheduler: bool = True
    
    # Logging
    verbose: int = 1
    log_dir: str = "logs"


# ==============================
# EVALUATION CONFIGURATION
# ==============================
@dataclass
class EvalConfig:
    """Evaluation and visualization configuration."""
    metrics: List[str] = field(default_factory=lambda: ['rmse', 'mae', 'mape', 'r2'])
    save_plots: bool = True
    plot_dpi: int = 150
    figsize: Tuple[int, int] = (14, 6)


# ==============================
# MASTER CONFIGURATION
# ==============================
@dataclass
class Config:
    """Master configuration combining all settings."""
    gpu: GPUConfig = field(default_factory=GPUConfig)
    data: DataConfig = field(default_factory=DataConfig)
    lstm: LSTMConfig = field(default_factory=LSTMConfig)
    gru: GRUConfig = field(default_factory=GRUConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    
    # Random seed for reproducibility
    seed: int = 42


def get_config() -> Config:
    """Get the default configuration."""
    return Config()


def print_config(config: Config):
    """Print configuration summary."""
    print("=" * 60)
    print(" CONFIGURATION SUMMARY")
    print("=" * 60)
    print(f"\n📁 Data Path: {DATA_PATH}")
    print(f"🎯 Target Commodity: {config.data.commodity}")
    print(f"📊 Sequence Length: {config.data.sequence_length}")
    print(f"\n🖥️  GPU Settings:")
    print(f"   Use GPU: {config.gpu.use_gpu}")
    print(f"   Mixed Precision (FP16): {config.gpu.mixed_precision}")
    print(f"   XLA Acceleration: {config.gpu.xla_acceleration}")
    print(f"\n🏋️ Training Settings:")
    print(f"   Epochs: {config.training.epochs}")
    print(f"   Batch Size: {config.training.batch_size}")
    print(f"   Early Stopping Patience: {config.training.patience}")
    print("=" * 60)
