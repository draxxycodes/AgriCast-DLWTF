# Agricultural Commodity Price Prediction System

## Deep Learning with TensorFlow - Major Assignment (CSE 3793)

An industry-grade intelligent system for predicting agricultural commodity prices using advanced deep learning architectures including LSTM, GRU, Transformer, and Ensemble models.

**Optimized for NVIDIA RTX 4060 with CUDA support.**

---

## 🎯 Problem Statement

Predict agricultural commodity prices using historical market data with deep learning models that capture long-term dependencies and complex patterns.

---

## 🏗️ Project Structure

```
DLWTF-Project/
├── src/
│   ├── main.py              # 🚀 Main entry point
│   ├── config.py            # ⚙️ Configuration (GPU, models, training)
│   ├── eda.py               # 📊 Exploratory Data Analysis
│   ├── data_loader.py       # 📁 Data loading & preprocessing
│   ├── feature_engineering.py
│   ├── training.py          # 🏋️ Training utilities
│   ├── evaluation.py        # 📈 Metrics & visualization
│   └── models/
│       ├── lstm_model.py    # LSTM with Attention
│       ├── gru_model.py     # Bidirectional GRU
│       ├── transformer_model.py
│       └── ensemble_model.py
├── data/                    # Place dataset here or use default path
├── models/                  # Saved model weights
├── outputs/
│   ├── figures/             # Visualizations
│   └── reports/             # Evaluation reports
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Full Pipeline

```bash
cd src
python main.py
```

### 3. Other Options

```bash
# Only EDA
python main.py --mode eda

# Only training
python main.py --mode train

# Train for specific commodity
python main.py --commodity Onion

# Train for Potato
python main.py --commodity Potato
```

---

## 🖥️ GPU Configuration

Optimized for **NVIDIA RTX 4060** with:

- ✅ CUDA acceleration
- ✅ Mixed Precision (FP16) for faster training
- ✅ XLA JIT compilation
- ✅ Memory growth enabled

Configuration can be modified in `src/config.py`.

---

## 🧠 Model Architectures

| Model | Key Features | Parameters |
|-------|--------------|------------|
| **LSTM** | Multi-Head Attention, 3 layers | ~500K |
| **GRU** | Bidirectional, Residual connections | ~400K |
| **Transformer** | 4 layers, 8 heads, Positional encoding | ~300K |
| **Ensemble** | Stacking meta-learner | Combines all |

---

## 📊 Dataset

Using: `/home/draxxy/Downloads/archive/Price_Agriculture_commodities_Week.csv`

- 23,094 records
- Features: State, District, Market, Commodity, Variety, Grade, Arrival_Date, Min/Max/Modal Price

---

## 📈 Evaluation Metrics

- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)
- R² Score (Coefficient of Determination)

---

## 👤 Author

Deep Learning with TensorFlow Project - CSE 3793
