# Setup Guide

## Prerequisites

- **Mac with Apple Silicon** (M1, M2, M3, M4)
- **macOS 12.0+** (preferably 13.0+)
- **Python 3.11 or 3.12** (3.12 recommended)
- **15+ GB free disk space** (for models and venv)

## Installation Steps

### 1. Clone Repository

git clone https://github.com/sydkwests/llm-inference-mac-m4-optimization.git
cd llm-inference-mac-m4-optimization

### 2. Create Virtual Environment

python3 -m venv venv
source venv/bin/activate

Verify Python version:
python --version # Should be 3.11+

### 3. Install Dependencies

pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

### 4. Verify Installation

python << 'PYEOF'
import mlx.core as mx
import mlx_lm
print("✅ MLX installed successfully")
PYEOF

## Quick Start

Run benchmark
PYTHONPATH=. python benchmarks/run_benchmark.py

Analyze results
python notebooks/analysis.py

View figures
open results/figures/

## Troubleshooting

### `ModuleNotFoundError: No module named 'mlx'`
source venv/bin/activate
pip install mlx mlx-lm --upgrade

### `numpy` build failure
pip install numpy>=1.26.0 --only-binary :all:

---

For detailed setup, see README.md
