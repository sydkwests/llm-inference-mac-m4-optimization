# Setup Guide

This document explains how to set up the environment and run the project on a Mac with Apple Silicon.

---

## 1. Prerequisites

- Mac with Apple Silicon (M1, M2, M3, or M4)
- macOS 12.0 or later
- Python 3.11 or 3.12 installed
- At least 15 GB of free disk space (for models + virtual environment)
- Git installed

Check Python:

python3 --version

---

## 2. Clone the Repository

git clone https://github.com/sydkwests/llm-inference-mac-m4-optimization.git
cd llm-inference-mac-m4-optimization

---

## 3. Create and Activate Virtual Environment

python3 -m venv venv
source venv/bin/activate
python --version # should show Python 3.11+ or 3.12

If you open a new terminal later:

cd /path/to/llm-inference-mac-m4-optimization
source venv/bin/activate

---

## 4. Install Dependencies

pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

This installs:

- `mlx`, `mlx-lm` – MLX inference framework for Apple Silicon
- `pandas`, `matplotlib`, `seaborn` – analysis and visualization
- `pyyaml`, `tqdm`, `psutil` – configuration and utilities

---

## 5. Verify MLX Installation

python << 'PYEOF'
import mlx.core as mx
import mlx_lm
print("✅ MLX and mlx-lm imported successfully")
PYEOF

If you see the ✅ message, your setup is correct.

---

## 6. Run a Quick Benchmark Test

From the project root with venv active:

PYTHONPATH=. python benchmarks/run_benchmark.py

This will:

- Run the benchmark grid for Llama 3.2‑1B (4-bit)
- Save raw JSON under `results/raw_results/`
- Print latency and throughput summary to the terminal

---

## 7. Run Full Analysis

python notebooks/analysis.py

This will:

- Load all JSON benchmark results
- Compute summary statistics (CSV files in `results/analysis/`)
- Generate figures (PNG files in `results/figures/`)

Open figures on macOS:

open results/figures

---

## 8. Troubleshooting

### Problem: `ModuleNotFoundError: No module named 'mlx'`

Make sure the virtual environment is activated:

source venv/bin/activate
pip install mlx mlx-lm --upgrade

### Problem: `numpy` build failure during install

Use pre-built wheels:

pip install --upgrade pip
pip install "numpy>=1.26.0" --only-binary :all:

### Problem: Benchmark is very slow the first time

This is expected:

- Models are downloaded on first use
- Caches are warmed up

Subsequent runs will be faster.

---

## 9. Next Steps

After setup:

- Read `README.md` for overall project description
- Use `benchmarks/run_benchmark.py` to run new experiments
- Use `notebooks/analysis.py` and `scripts/analyze_results.py` for analysis