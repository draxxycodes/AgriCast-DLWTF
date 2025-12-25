# 🌾 AgriCast: Agricultural Commodity Price Prediction System

## Deep Learning with TensorFlow - Major Assignment (CSE 3793)

An industry-grade intelligent system for predicting agricultural commodity prices using **10 advanced deep learning architectures** including LSTM, GRU, Transformer, TCN, WaveNet, N-BEATS, TFT, ConvLSTM, DenseNN, and Attention models.

**Optimized for NVIDIA RTX 4060 with CUDA support.**

---

## 🎯 Problem Statement

Predict agricultural commodity prices using historical market data with deep learning models that capture long-term dependencies, seasonal patterns, and complex market dynamics.

---

## 🏗️ Project Structure

```
AgriCast-DLWTF/
├── src/
│   ├── main.py                 # 🚀 Main entry point
│   ├── config.py               # ⚙️ Configuration (GPU, models, training)
│   ├── train_all.py            # 🏋️ Train all 10 models
│   ├── train_hybrid.py         # 🔀 Hybrid ensemble training
│   ├── combine_all.py          # 📊 Combine results & generate reports
│   ├── eda.py                  # 📉 Exploratory Data Analysis
│   ├── data_loader.py          # 📁 Data loading & preprocessing
│   ├── feature_engineering.py  # 🛠️ Feature engineering pipeline
│   ├── training.py             # 🏋️ Training utilities
│   ├── evaluation.py           # 📈 Metrics & visualization
│   ├── inference.py            # 🔮 Model inference
│   ├── prepare_data.py         # 📦 Data preparation utilities
│   ├── fetch_kaggle_data.py    # ⬇️ Kaggle dataset downloader
│   └── models/
│       ├── lstm_model.py       # LSTM with Multi-Head Attention
│       ├── gru_model.py        # Bidirectional GRU with Residuals
│       ├── transformer_model.py# Transformer with Positional Encoding
│       ├── tcn_model.py        # Temporal Convolutional Network
│       ├── wavenet_model.py    # WaveNet Architecture
│       ├── nbeats_model.py     # N-BEATS (Neural Basis Expansion)
│       ├── temporal_fusion.py  # Temporal Fusion Transformer
│       ├── ensemble_model.py   # Stacking Meta-Learner Ensemble
│       └── base_model.py       # Base model interface
├── data/                       # Dataset (download via Kaggle)
├── models/                     # Saved model weights (.keras)
├── outputs/
│   ├── figures/
│   │   ├── <model_name>/       # Individual model visualizations
│   │   └── comparison/         # Cross-model comparison charts
│   └── reports/
│       └── all_models_results.csv
├── notebook/                   # Jupyter notebooks
├── requirements.txt
└── .gitignore
```

---

## 🧠 Model Architectures (10 Models)

| Model | Architecture | Key Features | Parameters |
|-------|-------------|--------------|------------|
| **TCN** | Temporal Convolutional Network | Dilated causal convolutions, residual blocks | 29.6M |
| **WaveNet** | Dilated Causal CNN | Multi-scale temporal patterns, skip connections | 32.1M |
| **GRU** | Bidirectional GRU | Gated recurrent units, residual connections | 68.0M |
| **Attention** | Self-Attention Network | Multi-head attention mechanism | 28.6M |
| **Transformer** | Full Transformer | Positional encoding, 4 layers, 8 heads | 50.8M |
| **LSTM** | Deep LSTM | Multi-head attention, 3 stacked layers | 40.6M |
| **TFT** | Temporal Fusion Transformer | Variable selection, interpretable attention | 14.2M |
| **ConvLSTM** | Convolutional LSTM | Spatial-temporal feature learning | 13.1M |
| **DenseNN** | Dense Neural Network | Fully connected layers with dropout | 14.6M |
| **N-BEATS** | Neural Basis Expansion | Interpretable time series decomposition | 29.6M |

---

## 📊 Model Performance Results

| Rank | Model | RMSE ↓ | MAE | MAPE | R² Score |
|------|-------|--------|-----|------|----------|
| 🥇 1 | **TCN** | **634.74** | 321.82 | 75.00% | 0.469 |
| 🥈 2 | WaveNet | 701.54 | 369.25 | 69.43% | 0.351 |
| 🥉 3 | GRU | 710.60 | 383.77 | 79.53% | 0.335 |
| 4 | Attention | 714.24 | 391.04 | 71.04% | 0.328 |
| 5 | Transformer | 721.62 | 357.90 | 77.46% | 0.314 |
| 6 | LSTM | 724.23 | 359.19 | 74.23% | 0.309 |
| 7 | TFT | 770.05 | 399.55 | 72.30% | 0.219 |
| 8 | ConvLSTM | 886.83 | 523.97 | 84.90% | -0.036 |
| 9 | DenseNN | 943.09 | 638.10 | 81.65% | -0.172 |
| 10 | N-BEATS | 1111.84 | 774.11 | 84.63% | -0.629 |

*Lower RMSE/MAE/MAPE is better. Higher R² is better.*

---

## 🚀 Quick Start

### 1. Clone & Install Dependencies

```bash
git clone https://github.com/draxxycodes/AgriCast-DLWTF.git
cd AgriCast-DLWTF
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download Dataset

```bash
cd src
python fetch_kaggle_data.py
```

Or manually download from [Kaggle](https://www.kaggle.com/datasets) and place in `data/`.

### 3. Train All Models

```bash
cd src
python train_all.py
```

This will:
- Train all 10 deep learning models
- Generate individual model visualizations in `outputs/figures/<model_name>/`
- Create comparison charts in `outputs/figures/comparison/`
- Save results to `outputs/reports/all_models_results.csv`

### 4. Other Training Options

```bash
# Run full pipeline with EDA
python main.py

# Only EDA
python main.py --mode eda

# Train specific model
python main.py --mode train --model lstm

# Train for specific commodity
python main.py --commodity Onion
```

---

## 🖥️ GPU Configuration

Optimized for **NVIDIA RTX 4060** with:

- ✅ CUDA acceleration
- ✅ Mixed Precision (FP16) for faster training
- ✅ XLA JIT compilation
- ✅ Memory growth enabled
- ✅ Gradient clipping to prevent NaN issues

Configuration can be modified in `src/config.py`.

---

## 📊 Dataset

**Source**: Indian Agriculture Commodity Price Dataset

- **Records**: 23,094 weekly price records
- **Features**:
  - State, District, Market
  - Commodity, Variety, Grade
  - Arrival_Date
  - Min/Max/Modal Price

---

## 📈 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **RMSE** | Root Mean Square Error - penalizes large errors |
| **MAE** | Mean Absolute Error - average prediction error |
| **MAPE** | Mean Absolute Percentage Error - scale-independent |
| **R²** | Coefficient of Determination - explained variance |

---

## 📁 Generated Outputs

### Per-Model Visualizations (`outputs/figures/<model_name>/`)
- `training_curves.png` - Loss/metric progression
- `predictions.png` - Actual vs Predicted prices
- `error_distribution.png` - Prediction error analysis

### Comparison Charts (`outputs/figures/comparison/`)
- `metrics_comparison.png` - Bar chart of all metrics
- `params_comparison.png` - Model parameter counts
- `predictions_overlay.png` - All model predictions overlaid
- `performance_scatter.png` - RMSE vs R² scatter plot
- `training_epochs.png` - Epochs to convergence
- `model_rankings.png` - Overall model rankings

---

## 🔧 Key Features

- **Multi-Model Training**: Train 10 different architectures with a single command
- **Automated Visualization**: Generate comprehensive visualizations automatically
- **GPU Acceleration**: Fully optimized for NVIDIA GPUs with mixed precision
- **Gradient Clipping**: Pre-configured to prevent NaN gradient issues
- **Modular Architecture**: Easy to add new models or modify existing ones
- **Comprehensive Metrics**: Multiple evaluation metrics for thorough analysis

---

## 📝 Requirements

- Python 3.10+
- TensorFlow 2.15+ with GPU support
- NVIDIA GPU with CUDA 12.x (recommended)
- See `requirements.txt` for full dependencies

---

## 👤 Author

**Deep Learning with TensorFlow Project - CSE 3793**

---

## 📄 License

This project is for educational purposes as part of the CSE 3793 course.
