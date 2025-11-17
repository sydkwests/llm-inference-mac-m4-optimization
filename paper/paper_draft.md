# Efficient Local LLM Inference on Apple Silicon: A Benchmarking Study

## 1. Introduction
[To be written]

## 2. Methodology
[Describe hardware (Mac Mini M4), models, MLX framework, quantization, prompts, and metrics.]

## 3. Experimental Setup
[Detail: Python 3.12, MLX 0.29.4, mlx-lm 0.28.3, CLI wrapper, config (prompts, token lengths, temperatures, runs).]

## 4. Results

### 4.1 Overall Performance of Llama 3.2-1B

Across all successful configurations on the Mac M4 system, the 4-bit quantized Llama 3.2-1B model achieved an average end-to-end latency of 2.68 seconds per generation with a standard deviation of 0.69 seconds, measured across multiple prompts, output lengths, and temperatures. The fastest configuration exhibited a latency of approximately 1.93 seconds, while the slowest reached 3.91 seconds, reflecting the expected increase in latency as the requested output length grows.

In terms of throughput, the same model sustained an average of 51.9 tokens per second with a standard deviation of 9.89 tokens per second over the full benchmark grid. Observed throughput ranged from roughly 40.5 tokens per second at shorter outputs to up to 69.4 tokens per second at longer outputs, indicating that the Mac M4 is capable of delivering interactive response times while still maintaining token generation speeds suitable for real-time local assistants and reasoning workloads.

### 4.2 Scaling with Output Length
[To be filled in using token_scaling.csv and 03_latency_vs_tokens.png.]

### 4.3 Temperature Effects
[To be filled in using temperature_effect.csv and 04_temperature_effect.png.]

## 5. Discussion
[Interpret results: trade-offs between speed and output length, practical implications for local inference.]

## 6. Conclusion
[Summarize findings and outline future work, e.g., more models, batch decoding, different quantization schemes.]

