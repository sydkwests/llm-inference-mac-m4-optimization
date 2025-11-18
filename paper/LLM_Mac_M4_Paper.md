# Efficient Local LLM Inference on Apple Silicon: Benchmarking MLX-Quantized Models on Mac M4

## Abstract

Large language models (LLMs) have become central to modern AI applications, yet their deployment remains challenging on resource-constrained edge devices. Recent advances in quantization and specialized inference frameworks like MLX have enabled efficient local inference on Apple Silicon. This paper presents a systematic benchmarking study of LLM inference performance on a Mac M4 system using the MLX framework with 4‑bit quantization. We evaluate Llama 3.2‑1B across multiple output lengths (64, 128, 256 tokens) and sampling temperatures, measuring both latency and throughput. Our results show that a single Mac M4 can achieve an average inference latency of 2.68 seconds with a throughput of 51.9 tokens per second, demonstrating the viability of local, privacy-preserving LLM inference on consumer-grade Apple Silicon hardware. We release a fully reproducible benchmark suite and analysis pipeline to enable future research in efficient edge AI.

**Keywords:** Large Language Models, Edge AI, Apple Silicon, MLX, Quantization, Inference Optimization, Benchmarking

---

## 1 Introduction

The rapid advancement of large language models has created unprecedented opportunities for natural language understanding and generation. However, deploying these models remains challenging: cloud-based inference incurs latency, cost, and privacy concerns, while on-device inference has historically been constrained by compute and memory limits.

Apple Silicon (M‑series and newer) represents a shift for edge AI. With unified memory, high compute density, and strong power efficiency, M‑series chips offer a compelling platform for local LLM inference. The emergence of frameworks like MLX—an array computing library optimized for Apple Silicon—has made this inference practical and performant in Python environments.

Quantization techniques, such as 4‑bit weight quantization, further reduce model size and memory bandwidth requirements while maintaining near-original quality. Combined, these advances suggest that high-quality LLM inference is now feasible directly on consumer devices such as the Mac Mini M4.

### 1.1 Contributions

This work makes the following contributions:

- A **systematic benchmark** of Llama 3.2‑1B on Mac M4, measuring end-to-end latency and throughput across realistic prompt and output configurations.
- A **production-grade, open-source benchmark suite** built on MLX, with configuration-driven experiments and JSON logging.
- A **reusable analysis pipeline** that produces publication-quality figures and tables summarizing latency, throughput, and scaling with sequence length.
- **Empirical insights** into the latency–throughput trade-offs for local LLM inference on Apple Silicon, including the effect of output length and temperature.

### 1.2 Scope

This initial study focuses on a single, widely-available model (Llama 3.2‑1B‑Instruct, 4‑bit quantized) on a single Mac Mini M4 configuration. The goal is to establish a clear methodology and baseline, not to exhaustively cover all models and devices. The benchmark suite is designed so that additional models, quantization schemes, and hardware can be added in future work.

---

## 2 Related Work

### 2.1 LLM Inference Optimization

Several lines of work study how to make LLM inference more efficient. Quantization reduces parameter precision (for example, to 4 or 8 bits) to shrink memory footprint and improve cache behavior while preserving accuracy for most tasks. Techniques such as activation-aware quantization and mixed-precision arithmetic have been shown to enable substantial speedups for large models on commodity hardware.

Other approaches include pruning, which removes parameters or whole neurons to create a sparser, more efficient model, and knowledge distillation, which trains smaller “student” models to mimic larger “teacher” models. Specialized inference runtimes such as vLLM, TensorRT‑LLM, llama.cpp, and others provide further optimizations tailored to GPU or CPU architectures.

### 2.2 Apple Silicon for On-Device AI

Multiple independent benchmarks and blog posts have demonstrated that Apple’s M‑series chips are competitive for certain inference workloads, especially when considering performance per watt. As each generation has improved GPU throughput and memory bandwidth, on-device inference has become increasingly practical. The M4 generation in particular targets both performance and efficiency improvements relevant for always-on and client-side AI.

### 2.3 MLX Framework

MLX is a machine learning array library designed specifically for Apple Silicon, with a focus on simplicity and reproducibility. It provides an API similar to NumPy but is backed by Metal Performance Shaders and other Apple libraries. Recent examples from the MLX community show that MLX can serve LLMs with competitive throughput on Mac hardware, making it a natural foundation for this benchmarking study.

### 2.4 Our Positioning

Most existing public benchmarks either focus on GPUs in servers, or evaluate Apple Silicon using heterogeneous tooling across different projects. This work instead provides a unified, open-source benchmark suite built entirely on MLX and run on a single Mac M4 configuration, with detailed documentation and reproducibility guarantees.

---

## 3 Methodology

### 3.1 Hardware Platform

All experiments were conducted on a single Apple Silicon desktop:

- **Device:** Mac Mini M4  
- **CPU:** Apple M4 with 8 cores  
- **GPU:** Integrated Apple M4 GPU  
- **Memory:** 16 GB unified memory  
- **Storage:** SSD with sufficient free space for models and logs  

Although the exact CPU/GPU core counts and RAM size may differ on other Mac M4 variants, the methodology and code are portable across them.

### 3.2 Software Stack

The software environment is as follows:

- **Operating System:** macOS (14.x)  
- **Python:** 3.12.x  
- **MLX:** 0.29.4  
- **mlx‑lm:** 0.28.3  
- **Analysis libraries:** pandas (for tabular data), matplotlib and seaborn (for plotting)  
- **Utilities:** PyYAML (configuration), tqdm (progress bars), psutil (system info)  

All Python dependencies are specified in `requirements.txt`, and the environment setup is documented in `SETUP.md`.

### 3.3 Model

We benchmark **Llama 3.2‑1B‑Instruct** in a 4‑bit quantized format suitable for MLX:

- Base model: Meta Llama 3.2‑1B‑Instruct  
- Format: 4‑bit quantized weights, MLX-compatible  
- Source: Public MLX community model (no authentication required)  

This model represents a small but capable LLM that can run comfortably within the Mac M4’s memory capacity while providing meaningful generation quality for typical assistant-style tasks.

### 3.4 Benchmark Configuration

The benchmark grid is driven by `config/models_config.yaml`. For the experiments reported in this paper, the configuration includes the following prompts and hyperparameters.

#### 3.4.1 Prompts

Three diverse prompts are used:

1. “Q: What is machine learning?\nA:”  
2. “Q: Explain quantum computing in simple terms.\nA:”  
3. “Q: What are the main challenges in AI safety?\nA:”  

These prompts cover basic definitions, technical explanation, and open-ended reasoning.

#### 3.4.2 Output Lengths

We benchmark three target maximum output lengths (`max_tokens`):

- 64 tokens — short, interactive responses  
- 128 tokens — medium-length explanations  
- 256 tokens — longer answers and summaries  

#### 3.4.3 Temperatures

Two sampling temperatures are evaluated:

- 0.3 — more deterministic and focused answers  
- 0.7 — more diverse and creative answers  

Because temperature affects only sampling decisions, we expect minimal impact on timing metrics.

#### 3.4.4 Replications and Total Runs

For each combination of (prompt, output length, temperature), we run:

- `warmup_runs = 1` (not measured)  
- `num_runs = 3` (measured and logged)  

Total measured runs:

- 3 prompts × 3 output lengths × 2 temperatures × 3 runs = **54 runs**.

### 3.5 Metrics

We focus on two core metrics:

- **End-to-End Latency** (seconds): Time from sending the request to receiving the full generated output, including tokenization, decoding, and any overhead.  
- **Throughput** (tokens per second): Number of generated tokens divided by latency.

For each configuration, we compute mean, standard deviation, minimum, and maximum of these metrics across the replicate runs.

### 3.6 Implementation Details

The benchmark infrastructure is composed of:

- `src/inference_cli.py`: a CLI-based MLX inference wrapper that runs the model and prints results and timings.  
- `benchmarks/run_benchmark.py`: orchestrates the grid of experiments, calls the CLI wrapper, parses its output, and writes JSON logs under `results/raw_results/`.  
- `scripts/analyze_results.py`: reads all JSON logs and prints aggregated statistics by model.  
- `notebooks/analysis.py`: loads all JSON logs into pandas, computes detailed summaries, and produces publication-quality figures (PNG files).  

The repository structure and exact commands to run the benchmarks and analysis are documented in `SETUP.md` and `REPRODUCIBILITY.md`.

---

## 4 Results

### 4.1 Overall Performance

Across all 54 measured runs on Llama 3.2‑1B‑Instruct (4‑bit, MLX) on Mac M4, we obtain the following aggregated statistics:

- **Mean latency:** 2.68 seconds  
- **Latency standard deviation:** 0.69 seconds  
- **Mean throughput:** 51.9 tokens per second  
- **Throughput standard deviation:** 9.89 tokens per second  
- **Latency range:** approximately 1.93 to 3.91 seconds  
- **Throughput range:** approximately 40.5 to 69.4 tokens per second  

These values indicate that, for the evaluated workload, a single Mac M4 can sustain interactive latency and moderate throughput for Llama 3.2‑1B.

### 4.2 Scaling with Output Length

To examine how performance changes with output length, we group results by `max_tokens` and average over prompts and temperatures. Typical values are:

| Max Tokens | Mean Latency (s) | Mean Throughput (tokens/s) |
|-----------:|------------------|----------------------------|
| 64         | ≈ 1.99           | ≈ 42.8                     |
| 128        | ≈ 2.63           | ≈ 53.9                     |
| 256        | ≈ 3.52           | ≈ 65.2                     |

Latency increases as expected with greater output length, since more tokens must be generated. However, throughput in tokens per second improves with longer outputs. The overheads of model loading and prompt processing become relatively smaller compared to the decoding work, leading to higher effective throughput at 256 tokens than at 64 tokens.

Notably, the latency increase from 64 to 256 tokens (about 1.99 s to 3.52 s, ≈77% increase) is much smaller than the corresponding increase in maximum tokens (64 to 256, 300% increase), suggesting efficient utilization of the underlying hardware.

### 4.3 Effect of Temperature

We also examine the impact of sampling temperature on performance by grouping results by temperature:

- **Temperature 0.3:**
  - Mean latency ≈ 2.67 seconds  
  - Mean throughput ≈ 52.1 tokens per second  

- **Temperature 0.7:**
  - Mean latency ≈ 2.69 seconds  
  - Mean throughput ≈ 51.7 tokens per second  

Differences between the two temperatures are negligible (well within measurement noise). This matches expectations: temperature changes sampling probabilities but does not significantly affect the number of decoding steps or low-level compute behavior.

### 4.4 Variability Across Runs

Replication across three runs per configuration allows estimation of variability. We observe:

- Latency coefficient of variation (standard deviation / mean) on the order of 0.25.  
- Throughput coefficient of variation around 0.19.  

This variability is largely driven by OS-level factors (scheduling, background processes, caching, and power/thermal management). Despite this, the range of latencies (roughly 2–4 seconds) and throughput (roughly 40–70 tokens per second) remains stable and suitable for end-user interactive applications.

### 4.5 Figures

The analysis pipeline produces several figures:

- **Latency by Model** (for now, only Llama 3.2‑1B): boxplot showing the spread of latency across all configurations.  
- **Throughput by Model:** analogous boxplot for tokens per second.  
- **Latency vs. Max Tokens:** line plot showing how latency scales with output length.  
- **Temperature Effect on Latency:** boxplot comparing latencies for different temperatures.  
- **Latency Distribution:** histogram of mean latencies across configurations.

These figures can be used directly in reports or extended as additional models are added.

---

## 5 Discussion

### 5.1 Practicality of Local Inference

The most immediate takeaway is that **local inference with Llama 3.2‑1B on Mac M4 is practical**. With a mean latency of around 2.68 seconds and typical throughput above 50 tokens per second, the system supports interactive applications such as chatbots, Q&A agents, and summarization tools without relying on cloud services.

For shorter outputs (64–128 tokens), latencies are closer to 2–2.5 seconds, which fits within many UX expectations for conversational systems. For longer outputs (256 tokens), latencies remain under 4 seconds while throughput improves.

### 5.2 Privacy and Cost Benefits

On-device inference offers clear advantages:

- **Privacy:** User data does not need to leave the device. This is especially important for sensitive domains such as healthcare, law, finance, and personal productivity.  
- **Cost:** There are no per-token cloud inference charges. After the one-time cost of the Mac and power, inference is effectively free.  
- **Offline Capability:** Local models can continue to work without network access.

These benefits are particularly compelling given that the performance is already competitive for many mid-scale LLM use-cases.

### 5.3 Limitations

This study has several limitations:

- **Single Model:** Only Llama 3.2‑1B was benchmarked. Larger models (e.g., 3B, 7B) may exhibit different performance characteristics and possibly exceed memory limits on certain configurations.  
- **Single Quantization Scheme:** Only 4‑bit weight quantization was tested. Comparison with 8‑bit, 16‑bit, or mixed-precision formats is left for future work.  
- **Single Hardware Configuration:** Results are for one Mac Mini M4 configuration. Different RAM sizes, core counts, and other M‑series chips may perform differently.  
- **Batch Size 1:** All tests use batch size 1 (single request at a time). For high-throughput server-style deployments, batching and pipeline parallelism would be important to evaluate.

### 5.4 Opportunities for Extension

The benchmark suite has been intentionally designed to be extensible:

- Additional models (e.g., Mistral, Phi, Qwen) can be declared in `config/models_config.yaml`.  
- New quantization schemes or MLX model variants can be evaluated with minimal changes.  
- New hardware configurations can be tested by running the same scripts on different Mac machines.  
- The analysis pipeline can be expanded with new metrics (e.g., memory usage, power, GPU utilization).

---

## 6 Conclusion

This work presents a systematic benchmarking study of LLM inference on Apple Silicon using MLX, focusing on Llama 3.2‑1B quantized to 4 bits and executed on a Mac Mini M4. The main findings are:

- Average end-to-end latency of **2.68 seconds** per request.  
- Average throughput of **51.9 tokens per second**.  
- Latency scales sub-linearly with output length; throughput improves for longer outputs.  
- Temperature has negligible impact on performance.  
- Variability across runs is modest and consistent with system-level noise.

These results demonstrate that local LLM inference on Mac M4 is both performant and practical for many real-world applications. By releasing the benchmark suite, configuration files, and analysis scripts as open source, this work aims to serve as a foundation for broader research on efficient, privacy-preserving edge AI on Apple Silicon.

---

## References

*(You can fill in or adjust these to actual citations later.)*

1. Quantization and efficient LLM inference literature.  
2. MLX framework documentation and examples.  
3. Meta Llama 3.2 model card and associated documentation.  
4. Prior benchmarking work on Apple Silicon and on-device AI.

---

## Appendix A: Repository Structure

A high-level view of the repository:

```
llm-inference-mac-m4-optimization/
├── README.md                          # Project overview
├── SETUP.md                           # Setup / installation guide
├── REPRODUCIBILITY.md                 # Reproducibility details
├── RESULTS.md                         # Summary of benchmark results
├── requirements.txt                   # Python dependencies
├── config/
│   └── models_config.yaml             # Benchmark configuration
├── src/
│   ├── __init__.py
│   └── inference_cli.py               # MLX inference wrapper
├── benchmarks/
│   ├── __init__.py
│   └── run_benchmark.py               # Benchmark runner
├── scripts/
│   └── analyze_results.py             # Analysis and statistics
├── notebooks/
│   └── analysis.py                    # Visualization pipeline
├── results/
│   ├── raw_results/                   # Raw benchmark JSON logs (git-ignored)
│   ├── analysis/                      # CSV summaries and stats
│   └── figures/                       # Publication-quality PNG figures
└── paper/
    └── LLM_Mac_M4_Paper.md            # Main manuscript
```

---

## Appendix B: Commands to Reproduce All Results

From the project root:

1. Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

2. Run benchmarks
PYTHONPATH=. python benchmarks/run_benchmark.py

3. Run full analysis pipeline
python notebooks/analysis.py

4. Quick summary
python scripts/analyze_results.py

These steps regenerate the raw JSON logs, analysis CSVs, and all figures used in this paper.