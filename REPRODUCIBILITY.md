# Reproducibility Guide

This document describes exactly how to reproduce the experiments and results in this repository.

---

## 1. Hardware

All main experiments were run on:

- Device: Mac Mini M4
- CPU: Apple M4 (8-core)
- GPU: Apple M4 integrated GPU
- Memory: 16 GB unified memory
- Storage: SSD with at least 50 GB free
- OS: macOS (version 14.x or later)

The code should also run on other Apple Silicon machines (M1, M2, M3) but performance will differ.

---

## 2. Software Environment

### 2.1 Python and Dependencies

- Python: 3.12.x
- Package manager: `pip`

Main Python packages:

- `mlx` (Apple Silicon ML library)
- `mlx-lm` (LLM utilities on MLX)
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `pyyaml`
- `tqdm`
- `psutil`

These are all installed via:

pip install -r requirements.txt

### 2.2 Repository Version

To reproduce the exact experiments, use the tagged version (example):

git clone https://github.com/sydkwests/llm-inference-mac-m4-optimization.git
cd llm-inference-mac-m4-optimization
git checkout main

(If you later add a `v1.0` tag, you can specify `git checkout v1.0` here.)

---

## 3. Configuration

All benchmark settings are controlled by a single YAML file:

- `config/models_config.yaml`

Typical content:

models:

id: "mlx-community/Llama-3.2-1B-Instruct-4bit"
name: "Llama 3.2-1B"

prompts:

"Q: What is machine learning?\nA:"

"Q: Explain quantum computing in simple terms.\nA:"

"Q: What are the main challenges in AI safety?\nA:"

benchmark_settings:
max_tokens:​
temperatures: [0.3, 0.7]
num_runs: 3
warmup_runs: 1

You can reproduce all reported results by using this exact config.

---

## 4. Running the Benchmarks

### 4.1 One-Time Setup

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

PYTHONPATH=. python benchmarks/run_benchmark.py

What this does:

- Iterates over all model × prompt × max_tokens × temperature combinations
- Runs `warmup_runs` unmeasured warmup calls
- Runs `num_runs` measured runs per configuration
- Records latency and throughput metrics
- Saves results as JSON in `results/raw_results/benchmark_*.json`

---

## 5. Analysis and Figures

To regenerate all tables and figures used in the paper:

python notebooks/analysis.py

This script will:

- Load all `benchmark_*.json` files from `results/raw_results/`
- Create:
  - `results/analysis/summary_stats.csv`
  - `results/analysis/token_scaling.csv`
- Generate figures:
  - `results/figures/01_latency_by_model.png`
  - `results/figures/02_throughput_by_model.png`
  - `results/figures/03_latency_vs_tokens.png`
  - `results/figures/04_temperature_effect.png`
  - `results/figures/05_latency_distribution.png`

The `scripts/analyze_results.py` script also provides a quick textual summary:

python scripts/analyze_results.py

---

## 6. Randomness and Determinism

These benchmarks **do not** rely on random seeds for the metrics:

- Latency and throughput are measured at the system level.
- Sampling temperature affects token choice, not speed.

Minor variation in results (std dev) is due to:

- OS scheduling
- Background processes
- Caching effects

You should expect small differences (a few hundred milliseconds) but not order-of-magnitude changes.

---

## 7. Expected Results

For Llama 3.2‑1B (4-bit, MLX) on Mac Mini M4, typical aggregated results:

- Mean latency: ≈ 2.68 seconds
- Latency std: ≈ 0.69 seconds
- Mean throughput: ≈ 51.9 tokens/sec
- Throughput std: ≈ 9.89 tokens/sec
- Latency range: ≈ 1.93 – 3.91 seconds
- Throughput range: ≈ 40.5 – 69.4 tokens/sec

If your results are in this ballpark, your reproduction matches the original environment.

---

## 8. Raw Data

Raw benchmark logs:

- Location: `results/raw_results/benchmark_YYYYMMDD_HHMMSS.json`
- Automatically created by `run_benchmark.py`
- These files are **git-ignored** to keep the repository light.

Each JSON has:

- `metadata`: start_time, end_time, duration, number of experiments, counts
- `results`: list of per-configuration records with metrics

---

## 9. Extending the Experiments

To extend or modify experiments while preserving reproducibility:

1. Create a copy of the config:

cp config/models_config.yaml config/models_config_experiment2.yaml

2. Edit it (e.g., different models, more tokens, other temperatures).

3. Run: PYTHONPATH=. python benchmarks/run_benchmark.py

4. Keep track of which config file and git commit hash you used.

---

## 10. Reproducibility Checklist

To fully reproduce the paper:

- [ ] Use a Mac with Apple Silicon (preferably M4, 16 GB RAM)
- [ ] Install Python 3.11 or 3.12
- [ ] Clone this repository and check out the correct commit/tag
- [ ] Create and activate a virtual environment
- [ ] Install dependencies from `requirements.txt`
- [ ] Use `config/models_config.yaml` without modification
- [ ] Run `PYTHONPATH=. python benchmarks/run_benchmark.py`
- [ ] Run `python notebooks/analysis.py`
- [ ] Confirm aggregated metrics match those in the paper within reasonable variation

If any of these steps differ, document the differences when you report or extend results.

---

If you find any issues reproducing the results, please open an issue on GitHub with:

- Hardware details
- OS version
- Python and package versions
- Terminal logs
- Screenshots if relevant