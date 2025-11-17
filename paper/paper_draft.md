# Efficient Local LLM Inference on Apple Silicon: A Benchmarking Study

## 1. Introduction
[To be written]

## 2. Methodology
[Describe hardware, models, benchmark protocol, metrics]

## 3. Experimental Setup
[Describe MLX, quantization, prompts, configs]

## 4. Results

### 4.1 Preliminary Latency and Throughput

We conducted an initial benchmark of Llama 3.2‑1B (4‑bit quantized, MLX format) on a Mac M4 system using a single prompt and 64-token outputs. Across two independent runs, the model achieved an average latency of 2.14 seconds per generation with a standard deviation of 0.27 seconds, corresponding to a latency range of 1.95–2.33 seconds. This indicates that the system exhibits stable and predictable per-request latency even in this early configuration.

In terms of throughput, Llama 3.2‑1B sustained an average of 41.8 tokens per second with a standard deviation of 3.77 tokens per second, and an observed range between 39.1 and 44.5 tokens per second. These values demonstrate that a 1B-parameter model, when quantized to 4-bit and executed through MLX on Mac M4, can comfortably deliver interactive latencies while maintaining a token throughput sufficient for real-time applications such as conversational assistants and on-device reasoning.



### 4.2 Scaling with Output Length
[Will be filled in after full grid]

## 5. Discussion
[To be written]

## 6. Conclusion
[To be written]
