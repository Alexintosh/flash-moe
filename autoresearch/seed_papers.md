# Seed Papers for Flash-MoE Optimization

Pre-curated papers relevant to our specific architecture (MoE + SSD streaming + Metal GPU + Apple Silicon). The autoresearch agent should search for and read these, plus discover new papers via WebSearch.

## Expert I/O & Offloading

1. **"LLM in a Flash: Efficient Large Language Model Inference with Limited Memory"**
   - Authors: Alizadeh et al. (Apple), 2023
   - URL: https://arxiv.org/abs/2312.11514
   - Relevance: THE foundational paper for our approach. Windowing strategy, row-column bundling for flash reads, preloading based on activation sparsity. We already use their core idea but may have missed specific techniques.

2. **"MoE-Infinity: Offloading-Efficient MoE Model Serving"**
   - Authors: Xue et al., 2024
   - URL: https://arxiv.org/abs/2401.14361
   - Relevance: Expert-level offloading with activation-aware prefetch. Their "expert popularity" tracking could improve our page cache hit rate.

3. **"Pre-gated MoE: An Algorithm-System Co-Design for Fast and Scalable Mixture-of-Expert Inference"**
   - Authors: Hwang et al., 2024
   - URL: https://arxiv.org/abs/2308.12066
   - Relevance: Predicts expert routing from the previous layer's output, enabling prefetch. Different from our temporal predictor (which failed at 25% accuracy).

4. **"PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU"**
   - Authors: Song et al., 2023
   - URL: https://arxiv.org/abs/2312.12456
   - Relevance: Hot/cold neuron partitioning between GPU and CPU. Our tiered quantization is related but PowerInfer's activation prediction is more sophisticated.

## GPU Kernel Optimization

5. **"AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration"**
   - Authors: Lin et al., 2023
   - URL: https://arxiv.org/abs/2306.00978
   - Relevance: Activation-aware scaling before quantization + optimized GEMV kernels. Their kernel tiling strategy may apply to our dequant matvec.

6. **"FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"**
   - Authors: Dao, 2023
   - URL: https://arxiv.org/abs/2307.08691
   - Relevance: Online softmax, warp specialization, better tiling. We implemented fused online softmax based on this but there may be Metal-specific optimizations we missed.

7. **"FlashDecoding++: Faster Large Language Model Inference with Asynchronous Softmax"**
   - Authors: Hong et al., 2024
   - URL: https://arxiv.org/abs/2311.01282
   - Relevance: Flat GEMV optimization for decode phase. Their "unified maximum" technique eliminates the softmax sync point.

8. **"QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving"**
   - Authors: Lin et al., 2024
   - URL: https://arxiv.org/abs/2405.04532
   - Relevance: W4A8 serving with progressive quantization. Their SmoothAttention technique for KV4 is directly applicable to our FP8 KV cache.

## Speculative & Predictive Techniques

9. **"Sequoia: Scalable and Robust Speculative Decoding"**
   - Authors: Chen et al., 2024
   - URL: https://arxiv.org/abs/2402.12374
   - Relevance: Speculative decoding for offloaded models. Our MTP test was break-even, but Sequoia's tree-based approach handles MoE differently.

10. **"Mixtral of Experts"**
    - Authors: Jiang et al. (Mistral AI), 2024
    - URL: https://arxiv.org/abs/2401.04088
    - Relevance: The MoE architecture paper. Details on expert routing behavior, load balancing, and routing statistics that inform prefetch strategies.

## Apple Silicon Specific

11. **"Accelerating Large Language Model Decoding with Speculative Sampling"**
    - Authors: Leviathan et al., 2022
    - URL: https://arxiv.org/abs/2302.01318
    - Relevance: Original speculative decoding paper. The acceptance/rejection scheme could work with our tiered approach.

12. **Apple Metal Best Practices Guide**
    - URL: https://developer.apple.com/library/archive/documentation/3DDrawing/Conceptual/MTLBestPracticesGuide/
    - Relevance: Official guidance on command buffer management, resource allocation patterns, and GPU occupancy on Apple Silicon.

## To Discover

The agent should search for papers published in 2025-2026 on:
- "mixture of experts inference optimization"
- "apple silicon gpu compute optimization"
- "SSD offloading large language models"
- "4-bit quantized matmul GPU kernel"
- "expert routing prediction"
