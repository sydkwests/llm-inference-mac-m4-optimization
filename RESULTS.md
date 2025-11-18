# Benchmark Results Summary

This document summarizes the main quantitative results from the LLM inference benchmarks on Mac M4.

---

## 1. Experiment Setup (Short)

- **Hardware**: Mac Mini M4 (8‑core CPU, Apple GPU, 16 GB unified memory)
- **OS**: macOS (14.x)
- **Model**: Llama 3.2‑1B‑Instruct, 4‑bit quantized (MLX format)
- **Framework**: MLX 0.29.4 + mlx‑lm 0.28.3
- **Config**:
  - 3 prompts
  - 3 output lengths: 64, 128, 256 tokens
  - 2 temperatures: 0.3, 0.7
  - 3 runs per configuration

Total: **54 successful inference runs**.

---

## 2. Overall Performance

Aggregated across all prompts, output lengths, and temperatures:

- **Mean latency**: 2.68 seconds
- **Latency standard deviation**: 0.69 seconds
- **Mean throughput**: 51.9 tokens per second
- **Throughput standard deviation**: 9.89 tokens per second
- **Latency range**: 1.93 – 3.91 seconds
- **Throughput range**: 40.5 – 69.4 tokens per second

These values come from the analysis script:

python scripts/analyze_results.py

---

## 3. Performance by Output Length

From `results/analysis/token_scaling.csv` and `03_latency_vs_tokens.png`, the behavior by max_tokens is:

| Output Length (max_tokens) | Mean Latency (s) | Mean Throughput (tok/s) | Notes |
|----------------------------|------------------|--------------------------|-------|
| 64                         | ≈ 1.99           | ≈ 42.8                   | Short, interactive responses |
| 128                        | ≈ 2.63           | ≈ 53.9                   | Medium-length explanations |
| 256                        | ≈ 3.52           | ≈ 65.2                   | Longer generations, best throughput |

**Observations:**

1. Latency increases as expected with longer outputs (more tokens to decode).
2. Throughput (tokens/sec) actually **improves** at longer outputs because fixed overhead (model load, prompt processing) is amortized.
3. Latency growth is **sub-linear** relative to token increase, which is a good sign for hardware efficiency.

---

## 4. Temperature Effects

Using `results/analysis/temperature_effect.csv` and `04_temperature_effect.png`:

- **Temperature 0.3**:
  - Latency: ≈ 2.67 s
  - Throughput: ≈ 52.1 tok/s
- **Temperature 0.7**:
  - Latency: ≈ 2.69 s
  - Throughput: ≈ 51.7 tok/s

**Conclusion**: Changing temperature from 0.3 to 0.7 has **negligible impact** on latency and throughput (differences are well within measurement noise). Temperature can be tuned for response quality without worrying about speed.

---

## 5. Variability and Stability

Across runs:

- Latency standard deviation is ≈ 0.69 s over a mean of 2.68 s.
- Throughput standard deviation is ≈ 9.89 tok/s over a mean of 51.9 tok/s.

This variability is expected due to:

- OS scheduling
- Background processes
- Interaction with macOS power/thermal management

Despite this, latencies consistently fall in the **2–4 second** range, and throughput stays within **≈ 40–70 tok/s**, which is stable enough for interactive applications.

---

## 6. Figures Produced

The analysis pipeline (`python notebooks/analysis.py`) produces:

- `results/figures/01_latency_by_model.png` – Latency distribution by model
- `results/figures/02_throughput_by_model.png` – Throughput distribution
- `results/figures/03_latency_vs_tokens.png` – Latency vs output length
- `results/figures/04_temperature_effect.png` – Temperature vs latency
- `results/figures/05_latency_distribution.png` – Histogram of latencies

These can be directly used in reports, blog posts, or the research paper.

---

## 7. Interpretation for Real Use

- **Interactive Chat**: With ≈ 2–3 s latency and ~40–50 tok/s throughput at 64–128 tokens, Llama 3.2‑1B on Mac M4 is well‑suited for interactive chat or Q&A.
- **Long-form Generation**: For 256‑token outputs, latency is still below 4 s while throughput improves to ~65 tok/s, making it efficient for summarization and explanation tasks.
- **On-device AI**: All inference is local, which removes cloud latency and cost, and preserves privacy for user data.

---

## 8. How to Regenerate These Results

To reproduce everything in this document:

1. Activate environment
source venv/bin/activate

2. Run benchmark
PYTHONPATH=. python benchmarks/run_benchmark.py

3. Run full analysis + figures
python notebooks/analysis.py

4. Quick textual summary
python scripts/analyze_results.py

The numbers you see should be very close to those reported here, with minor variations due to system noise.

---

For deeper details and discussion, see the full paper in `paper/` and the reproducibility details in `REPRODUCIBILITY.md`.
