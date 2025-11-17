# Efficient Local LLM Inference on Apple Silicon: Mac M4 Benchmarking Study

A systematic benchmarking framework for evaluating large language model (LLM) inference performance on Apple Silicon using MLX and 4-bit quantization.

## 🚀 Quick Start

Clone
git clone https://github.com/sydkwests/llm-inference-mac-m4-optimization.git
cd llm-inference-mac-m4-optimization

Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

Run benchmark
PYTHONPATH=. python benchmarks/run_benchmark.py

Analyze
python notebooks/analysis.py

## 📊 Key Results

| Metric | Value |
|--------|-------|
| **Model** | Llama 3.2-1B (4-bit, MLX) |
| **Hardware** | Mac Mini M4 |
| **Mean Latency** | 2.68 ± 0.69 seconds |
| **Mean Throughput** | 51.9 ± 9.89 tokens/sec |
| **Experiments** | 54 successful runs |

## 📁 What's Inside

- `paper/` - Full research paper (10 pages)
- `src/` - MLX inference wrapper
- `benchmarks/` - Benchmark runner
- `scripts/` - Analysis tools
- `config/` - Configuration files
- `results/` - Benchmark outputs + figures

## 📄 Documentation

- `README.md` (this file) - Overview
- `INSTALLATION.md` - Setup guide
- `BENCHMARKING.md` - How to run benchmarks
- `RESULTS.md` - Key findings
- `FUTURE_WORK.md` - Research roadmap

## 📞 Contact

- GitHub: https://github.com/sydkwests/llm-inference-mac-m4-optimization
- Questions? Open an issue!

---

**Status**: ✅ Ready for publication
