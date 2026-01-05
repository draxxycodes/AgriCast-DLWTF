#!/bin/bash
# Activate virtual environment
source /home/draxxy/Documents/DLWTF-Project/venv/bin/activate

# Go to source directory
cd /home/draxxy/Documents/DLWTF-Project/src

echo "🚀 Starting Sequential Training..."
echo "=================================="

# 1. WaveNet (Retrying)
echo "🌊 Training WaveNet..."
python train_all.py --model WaveNet
echo "✓ WaveNet Done (or Failed)"

# 2. TCN
echo "📡 Training TCN..."
python train_all.py --model TCN
echo "✓ TCN Done"

# 3. PatchTST
echo "🧩 Training PatchTST..."
python train_all.py --model PatchTST
echo "✓ PatchTST Done"

# 4. LSTM
echo "🧠 Training LSTM..."
python train_all.py --model LSTM
echo "✓ LSTM Done"

# 5. GRU
echo "⚡ Training GRU..."
python train_all.py --model GRU
echo "✓ GRU Done"

# 6. Transformer
echo "🤖 Training Transformer..."
python train_all.py --model Transformer
echo "✓ Transformer Done"

echo "=================================="
echo "🎉 All Models Processed!"
