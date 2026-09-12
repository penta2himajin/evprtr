---
license: mit
language: en
library_name: transformers
pipeline_tag: text-generation
tags:
  - causal-lm
  - mixture-of-experts
  - reasoning
  - ternary
  - custom-code
---

# Maple-Preview

**DeepGrove · 2026**

Today we introduce Maple-Preview, an open-source 20B-A1B ternary-weight reasoning LLM. Maple-Preview has SOTA reasoning for its weight class and is even competitive with larger models. It solves IMO-level problems and runs at 200+ tokens/sec on a Mac mini M4, 5–16× faster than efficient models like Gemma 4, Qwen3.5, and gpt-oss.

- 20B-A1B Model
- 218 tok/s M4 Mac mini
- 5.31 GB Checkpoint
- 131,072 Token context

![Maple-Preview speed and performance frontier](assets/01-speed-frontier.png)

> [!NOTE]
> The included Transformers implementation depends on Triton and FlashAttention
> and is intended for a compatible CUDA environment. The reported Apple Silicon
> result uses a separate on-device runtime.

## Architecture

Maple-Preview is a 20B-A1B reasoning model designed from the start for efficient on-device inference. It utilizes a 24-layer, 256-expert (8 active) configuration with 3:1 SWA-512:GA attention. 

## Evaluation

On benchmarks, Maple-Preview sets a new point on the Pareto frontier for both memory-to-performance and speed-to-performance, demonstrating its strong reasoning capabilities. However, we note that this preview is focused primarily on raw reasoning and, as such, may underperform on agentic benchmarks. We intend to continue improving general performance through extended training before Maple's full release.

![Benchmark score comparison](assets/05-benchmark-scores-table.png)

Capability comparison using the dense output head across LCBv6, AIME 2026, HMMT 2026, and GPQA-D.

## Limitations

This preview received minimal post-training for agentic tasks and only
small-scale general reinforcement learning.

## License

Maple-Preview is released under the [MIT License](LICENSE).
