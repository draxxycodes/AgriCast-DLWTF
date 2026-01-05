# 🌾 AgriCast: PPT Presentation Plan (20 Slides)

**Course:** Deep Learning with TensorFlow - CSE 3793  
**Git Commit:** `370dea75` - All 10 model architectures with visualizations

---

## Slide 1: Title Slide

| Element | Content |
|---------|---------|
| **Title** | 🌾 AgriCast: Agricultural Commodity Price Prediction |
| **Subtitle** | Deep Learning with TensorFlow - CSE 3793 Major Assignment |
| **Key Stats** | 10 Deep Learning Architectures • 350M+ Parameters • Best RMSE: 634.74 |
| **Visual** | `outputs/figures/comparison/05_radar_chart.png` |

---

## Slide 2: Project Overview & Objectives

| Element | Content |`
|---------|---------|
| **Title** | Project Overview |
| **Content** | |
| | • Industry-grade intelligent system for agricultural commodity price prediction |
| | • Uses 10 advanced deep learning architectures (LSTM, GRU, Transformer, TCN, WaveNet, N-BEATS, TFT, ConvLSTM, DenseNN, Attention) |
| | • 350+ million total trainable parameters |
| | • GPU-optimized with mixed precision training on RTX 4060 |

---

## Slide 3: 🏆 Model Performance Leaderboard

| Rank | Model | RMSE ↓ | R² Score | Parameters |
|:----:|:------|-------:|---------:|-----------:|
| 🥇 | **TCN** | **634.74** | **0.469** | 29.6M |
| 🥈 | WaveNet | 701.54 | 0.351 | 32.1M |
| 🥉 | GRU | 710.60 | 0.335 | 68.0M |
| 4 | Attention | 714.24 | 0.328 | 28.6M |
| 5 | Transformer | 721.62 | 0.314 | 50.8M |
| 6 | LSTM | 724.23 | 0.309 | 40.6M |
| 7 | TFT | 770.05 | 0.219 | 14.2M |
| 8 | ConvLSTM | 886.83 | -0.036 | 13.1M |
| 9 | DenseNN | 943.09 | -0.172 | 14.6M |
| 10 | N-BEATS | 1111.84 | -0.629 | 29.6M |

**Figure:** `outputs/figures/comparison/01_metrics_bars.png`

---

## Slide 4: Dataset - Multi-Source Compilation

| Source | Dataset | Records | Coverage |
|--------|---------|--------:|----------|
| data.gov.in | Agriculture Commodities | 23,094 | India - Weekly |
| Kaggle | WFP India Food Prices | ~15,000 | India - UN WFP |
| Kaggle | Vegetables & Fruits | ~8,000 | Nepal - Kalimati |
| Kaggle | WFP Global Food Prices | ~50,000 | Global - 80+ countries |
| Kaggle | Commodity Prices 1960-2021 | ~3,000 | Global - Historical |
| Kaggle | Crop Price Prediction | ~2,000 | India - Crop yields |

**Final Dataset:** 7,015 daily records • 32 years (1992-2024) • 220 KB

---

## Slide 5: Data Processing & Feature Engineering

**Pipeline:** Raw Data → Standardization → Cleaning → Daily Aggregation → Final Dataset

| Feature | Type | Description |
|---------|------|-------------|
| price, log_price | Raw/Transform | Original and log-transformed |
| pct_change | Derived | Day-over-day change |
| ma_7, ma_14, ma_30 | Rolling | Moving averages |
| std_7, std_14, std_30 | Rolling | Standard deviations |
| momentum | Derived | Price deviation from MA_7 |

**Data Split:** Train 70% (4,910) • Val 15% (1,052) • Test 15% (1,053) • Sequence: 60 days

---

## Slide 6: 🥇 TCN - Best Performing Model

**RMSE: 634.74 | R²: 0.469 | Params: 29.6M**

```
Input (60×10) → Conv1D(512) → 18× Dilated Causal Blocks → GAP → Dense → Output
```

**Key Features:**
- Dilated convolutions capture long-range dependencies
- Causal padding prevents future information leakage
- Residual connections enable very deep networks

**Figures:** `tcn/predictions.png` + `tcn/training_curves.png`

---

## Slide 7: 🥈 WaveNet & 🥉 GRU

**WaveNet** (RMSE: 701.54 | R²: 0.351 | 32.1M)
```
Input → Conv1D → 22× Gated Dilated Blocks → Skip Aggregation → Dense → Output
```
- Gated activations from audio synthesis
- Skip connections aggregate multi-scale patterns

**GRU** (RMSE: 710.60 | R²: 0.335 | 68.0M)
```
Input → Dense(1024) → 8× BiGRU Residual Blocks → Attention(16h) → Dense → Output
```
- Bidirectional processing captures past/future context
- Deep residual connections prevent vanishing gradients

**Figures:** `wavenet/predictions.png` + `gru/predictions.png`

---

## Slide 8: Attention & Transformer Models

**Attention** (RMSE: 714.24 | R²: 0.328 | 28.6M)
```
Input → Embedding(384) → Positional Enc → 12× Pre-Norm Attention Blocks → Output
```

**Transformer** (RMSE: 721.62 | R²: 0.314 | 50.8M)
```
Input → Embedding(512) → Positional Enc → 12× Pre-Norm Transformer Blocks → Output
```

**Key Features:**
- Pre-normalization ensures gradient stability
- Fully parallelizable (no recurrence)
- 16 heads, key_dim=64

**Figures:** `attention/predictions.png` + `transformer/predictions.png`

---

## Slide 9: LSTM & TFT Models

**LSTM** (RMSE: 724.23 | R²: 0.309 | 40.6M)
```
Input → Dense(768) → 5× BiLSTM → 2× Multi-Head Attention → Dense → Output
```
- Classic LSTM with dual attention layers

**TFT** (RMSE: 770.05 | R²: 0.219 | 14.2M)
```
Input → Variable Selection → 3× BiLSTM → 2× Attention → Gated Skip → Output
```
- Variable selection learns feature importance
- **Most efficient** (best R²/params ratio)

**Figures:** `lstm/predictions.png` + `tft/predictions.png`

---

## Slide 10: ConvLSTM, DenseNN & N-BEATS

**ConvLSTM** (RMSE: 886.83 | Params: 13.1M)
- 6× Conv1D → MaxPool → 4× BiLSTM → Dense
- CNN extracts local patterns, LSTM captures temporal

**DenseNN** (RMSE: 943.09 | Params: 14.6M)
- Flatten → 11× Dense Layers (2048→128) with GELU, Dropout
- Pure MLP baseline, no temporal inductive bias

**N-BEATS** (RMSE: 1111.84 | Params: 29.6M)
- 12× N-BEATS Blocks with Backcast/Forecast branches
- Interpretable time series decomposition

**Figures:** `convlstm/predictions.png` + `densenn/predictions.png` + `nbeats/predictions.png`

---

## Slide 11: All Model Predictions vs Actual

**Visual:** `outputs/figures/comparison/03_predictions_overlay.png` (FULL WIDTH)

Overlay comparison of all 10 models against actual price data showing:
- TCN tracks actual values most closely
- N-BEATS shows highest deviation
- Top 6 models cluster together in performance

---

## Slide 12: Performance Scatter (RMSE vs R²)

**Visual:** `outputs/figures/comparison/04_performance_scatter.png`

**Insights:**
- TCN occupies optimal position (low RMSE, high R²)
- WaveNet, GRU, Attention, Transformer, LSTM cluster together
- ConvLSTM, DenseNN, N-BEATS show negative R² (worse than baseline)

---

## Slide 13: Model Comparison Charts

**Grid Layout (2×2):**

| Chart | Figure |
|-------|--------|
| Metrics Bar Comparison | `comparison/01_metrics_bars.png` |
| Parameters & Epochs | `comparison/02_params_epochs.png` |
| Radar Chart | `comparison/05_radar_chart.png` |
| Performance Heatmap | `comparison/06_heatmap.png` |

---

## Slide 14: Error Analysis

**Grid Layout (2×2):**

| Chart | Figure |
|-------|--------|
| Error Box Plots | `comparison/07_error_boxplots.png` |
| Learning Curves | `comparison/08_learning_curves.png` |
| Residual Plots | `comparison/09_residual_plots.png` |
| Cumulative Error | `comparison/10_cumulative_error.png` |

---

## Slide 15: Efficiency Analysis

**Visual:** `outputs/figures/comparison/11_efficiency_plot.png`

| Insight | Detail |
|---------|--------|
| Most Efficient | TFT - Best R² per parameter |
| Least Efficient | GRU - 68M params, moderate R² |
| Best Overall | TCN - Optimal balance |

---

## Slide 16: GPU Configuration & Optimizations

| Feature | Setting |
|---------|---------|
| Hardware | NVIDIA RTX 4060 (8GB VRAM) |
| CUDA | ✅ Enabled |
| Mixed Precision | ✅ FP16 (2x faster, 50% less memory) |
| XLA JIT | ✅ Enabled |
| Gradient Clipping | clipnorm=1.0 |
| Memory Growth | Dynamic |

**Training Config:** AdamW optimizer • Model-specific LR (5e-5 to 1e-4) • Early stopping (patience=35) • Batch size: 32

---

## Slide 17: Key Technical Features

| Category | Features |
|----------|----------|
| **Training Stability** | Pre-LayerNorm for Transformer/Attention, Gradient clipping, Huber loss |
| **Performance** | Mixed precision (FP16), Early stopping, LR reduction on plateau |
| **Visualization** | 12 comparison charts, Per-model curves, 200 DPI quality |

---

## Slide 18: Evaluation Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **RMSE** | √(Σ(y-ŷ)²/n) | Penalizes large errors |
| **MAE** | Σ\|y-ŷ\|/n | Average absolute error |
| **MAPE** | 100×Σ\|(y-ŷ)/y\|/n | Scale-independent % |
| **R²** | 1 - SS_res/SS_tot | Explained variance |

---

## Slide 19: Conclusion & Key Findings

| Finding | Detail |
|---------|--------|
| ✅ Best Model | TCN (RMSE: 634.74, R²: 0.469) |
| ✅ Convolutional Wins | TCN, WaveNet outperform RNNs |
| ✅ Pre-Norm Critical | Essential for Transformer stability |
| ✅ Efficiency | TFT achieves most with fewest params |

**Why TCN Wins:**
1. Causal convolutions respect temporal order
2. Dilated layers capture long-range dependencies
3. Fully parallelizable (faster than RNNs)
4. Residual connections enable deep networks

---

## Slide 20: Thank You / Q&A

| Element | Content |
|---------|---------|
| **Title** | Thank You |
| **Subtitle** | Questions & Discussion |
| **Course** | Deep Learning with TensorFlow - CSE 3793 |
| **Visual** | `outputs/figures/comparison/05_radar_chart.png` |

---

## 📊 Complete Figure List (42 images)

### Individual Model Figures (30 images)
| Model | Files |
|-------|-------|
| TCN | `tcn/predictions.png`, `tcn/training_curves.png`, `tcn/error_analysis.png` |
| WaveNet | `wavenet/predictions.png`, `wavenet/training_curves.png`, `wavenet/error_analysis.png` |
| GRU | `gru/predictions.png`, `gru/training_curves.png`, `gru/error_analysis.png` |
| Attention | `attention/predictions.png`, `attention/training_curves.png`, `attention/error_analysis.png` |
| Transformer | `transformer/predictions.png`, `transformer/training_curves.png`, `transformer/error_analysis.png` |
| LSTM | `lstm/predictions.png`, `lstm/training_curves.png`, `lstm/error_analysis.png` |
| TFT | `tft/predictions.png`, `tft/training_curves.png`, `tft/error_analysis.png` |
| ConvLSTM | `convlstm/predictions.png`, `convlstm/training_curves.png`, `convlstm/error_analysis.png` |
| DenseNN | `densenn/predictions.png`, `densenn/training_curves.png`, `densenn/error_analysis.png` |
| N-BEATS | `nbeats/predictions.png`, `nbeats/training_curves.png`, `nbeats/error_analysis.png` |

### Comparison Figures (12 images)
| # | File | Suggested Slide |
|:-:|------|-----------------|
| 1 | `01_metrics_bars.png` | Slides 3, 13 |
| 2 | `02_params_epochs.png` | Slide 13 |
| 3 | `03_predictions_overlay.png` | Slide 11 |
| 4 | `04_performance_scatter.png` | Slide 12 |
| 5 | `05_radar_chart.png` | Slides 1, 13, 20 |
| 6 | `06_heatmap.png` | Slide 13 |
| 7 | `07_error_boxplots.png` | Slide 14 |
| 8 | `08_learning_curves.png` | Slide 14 |
| 9 | `09_residual_plots.png` | Slide 14 |
| 10 | `10_cumulative_error.png` | Slide 14 |
| 11 | `11_efficiency_plot.png` | Slide 15 |
| 12 | `12_mae_vs_rmse.png` | Slide 14 |

*All paths relative to `outputs/figures/`*
