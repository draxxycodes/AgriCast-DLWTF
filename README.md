<h1 align="center">🌾 AgriCast: Agricultural Commodity Price Prediction</h1>

<p align="center">
  <b>Deep Learning with TensorFlow - CSE 3793 Major Assignment</b>
</p>

<p align="center">
  <a href="#-quick-start"><img src="https://img.shields.io/badge/Quick%20Start-Guide-blue?style=for-the-badge" alt="Quick Start"/></a>
  <a href="#-model-architectures"><img src="https://img.shields.io/badge/Models-7%20Architectures-green?style=for-the-badge" alt="Models"/></a>
  <a href="#-results"><img src="https://img.shields.io/badge/Best%20R²-0.32-orange?style=for-the-badge" alt="Best Score"/></a>
  <a href="#-gpu-configuration"><img src="https://img.shields.io/badge/GPU-RTX%204060-red?style=for-the-badge" alt="GPU"/></a>
</p>

<p align="center">
  An <b>industry-grade intelligent system</b> for predicting agricultural commodity prices using <b>7 advanced deep learning architectures</b> (PatchTST, N-BEATS, WaveNet, TCN, Transformer, GRU, LSTM) optimized as <b>Tiny Versions</b> for high-efficiency training on large tabular datasets (~827k records).
</p>

<p align="center">
  <img src="outputs/figures/comparison/05_radar_chart.png" alt="Model Performance Radar Chart" width="600"/>
</p>

---

## 📊 Performance Overview

### 🏆 Model Leaderboard

| Rank | Model | RMSE ↓ | MAE | Accuracy | R² Score | Parameters |
|:----:|:------|-------:|----:|---------:|---------:|-----------:|
| 🥇 | **PatchTST** | **0.612** | **0.445** | **78.5%** | **0.321** | 1.1M |
| 🥈 | **N-BEATS** | 0.625 | 0.458 | 76.2% | 0.294 | 17.5M |
| 🥉 | **Transformer** | 0.631 | 0.462 | 75.8% | 0.285 | 2.1M |
| 4 | WaveNet | 0.645 | 0.475 | 74.9% | 0.254 | 0.6M |
| 5 | TCN | 0.652 | 0.481 | 73.5% | 0.241 | 0.5M |
| 6 | GRU | 0.668 | 0.495 | 71.2% | 0.215 | 1.8M |
| 7 | LSTM | 0.675 | 0.502 | 70.1% | 0.195 | 1.9M |

> **📈 Best Overall**: PatchTST achieves the higher R² (0.321) and Directional Accuracy (78.5%).
> **⚡ Most Efficient**: TCN achieves competitive results with only ~0.5M parameters.

<p align="center">
  <img src="outputs/figures/comparison/01_metrics_bars.png" alt="Model Metrics Comparison" width="100%"/>
</p>

---

## 🧠 Model Architectures (Tiny & Optimized)

We engineered **"Tiny Versions" (<2M parameters)** of state-of-the-art architectures to prevent overfitting and maximize training speed.

### 1. PatchTST (2023 SOTA)
*   **Concept**: Treats time series as distinct channels (Channel Independence) and patches them like Vision Transformers.
*   **Key Tech**: RevIN (Reversible Normalization) + Conv1D Patching.
*   **Why it wins**: Captures local semantic patterns while maintaining global context.

### 2. N-BEATS
*   **Concept**: Pure Deep Learning architecture with no convolutions or recurrence.
*   **Key Tech**: Stack of blocks for Trend (polynomial) and Seasonality (Fourier).
*   **Why it wins**: Interpretable decomposition of the signal.

### 3. WaveNet
*   **Concept**: Adapted from DeepMind's audio generation model.
*   **Key Tech**: **Gated Activations** (`tanh * sigmoid`) acting as information filters.
*   **Why it wins**: Filters out noise very effectively.

### 4. TCN (Temporal Convolutional Network)
*   **Concept**: "ResNet for Time Series".
*   **Key Tech**: Causal Dilated Convolutions + Residual Connections + Spatial Dropout.
*   **Why it wins**: Incredible stability and huge receptive field (125 steps).

### 5. Transformer
*   **Concept**: Physics-aware Attention mechanism.
*   **Key Tech**: Pre-LayerNorm config + Multi-Head Self-Attention.
*   **Why it wins**: Finds correlations between distant time points.

*(Legacy models like ConvLSTM/TFT were evaluated and removed due to lower efficiency).*

---

## 📦 Dataset: Processed Agricultural Data

We processed a massive multi-source dataset specifically for this project.

| Property | Value |
|----------|-------|
| **File** | `data/processed_agricultural.csv` |
| **Records** | **827,014** total records |
| **Features** | **33** engineered features |
| **Commodities** | 445 distinct agricultural products |
| **Date Range** | 1992 - 2024 (32 Years) |

### Feature Engineering
*   **Rolling Stats**: 7, 14, 30-day Means and Std Dev.
*   **Cyclical**: Day of week / Month encoded as Sine/Cosine.
*   **Target**: Log-Returns (Stationary) + Robust Scaling.

---

## 🚀 Quick Start

### 1. Setup Environment
```bash
git clone https://github.com/draxxycodes/AgriCast-DLWTF.git
cd AgriCast-DLWTF
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Train Models
All training is handled by the **optimized** `src/train_all.py` script.

```bash
cd src

# Option A: Train models individually (Recommended)
python train_all.py --model PatchTST
python train_all.py --model WaveNet

# Option B: Run full sequence
python train_all.py --all

# Option C: Generate Comparison Charts (After training)
python train_all.py --compare
```

---

## 🔧 Technical Implementation Details

*   **Mixed Precision**: FP16 enabled for 2x speedup on RTX 4060.
*   **Optimization**: AdamW with Gradient Clipping (`clipnorm=1.0`) to prevent exploding gradients.
*   **Scheduling**: ReduceLROnPlateau (Start: 1e-3 -> Min: 1e-7).
*   **Evaluation**: Custom metrics including **Directional Accuracy** (Up/Down prediction) and **Information Coefficient**.

---

## 👤 Author
**Deep Learning with TensorFlow Project - CSE 3793**

---
<p align="center">Made with ❤️ using TensorFlow 2.15 & Keras 3</p>
