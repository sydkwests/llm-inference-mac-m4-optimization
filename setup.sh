#!/bin/bash
set -e

echo "🚀 Setting up LLM Inference Optimization project..."

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install requirements
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Create directories if they don't exist
echo "📁 Creating project directories..."
mkdir -p results/raw_results results/analysis results/figures

echo "✅ Setup complete!"
echo "📝 Next step: source venv/bin/activate"
