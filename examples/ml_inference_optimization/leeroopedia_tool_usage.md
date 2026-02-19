# Leeroopedia Knowledge Base — Tool Usage Reference

This document describes the Leeroopedia MCP tools available during the
with-KB benchmark run. It is a standalone reference and is **not** fed
to the agents automatically.

---

## Leeroopedia Knowledge Base (MANDATORY)

You have access to the Leeroopedia MCP tools. The KB contains production-grade CUDA and Triton kernel implementations from TransformerEngine, DeepSpeed, vLLM, Bitsandbytes, ggml, Ncnn, and MNN.

**You MUST use these tools as part of your workflow for every problem.** Do not rely solely on your training knowledge — the KB contains hardware-specific optimization patterns, tested block sizes, and proven kernel designs that will produce better results than writing kernels from scratch.

### Required Per-Problem Workflow

For **each** of the 10 problems, follow this sequence:

1. **Search** — Call `search_knowledge` with the problem's operation, tensor shapes, and target GPU. Example:
   ```
   search_knowledge("fused LayerNorm + GELU kernel for 5D tensor (32, 64, 32, 64, 64), reducing over last dim=64, NVIDIA L4 Ada Lovelace")
   ```

2. **Hypothesize** — Call `propose_hypothesis` with 2-3 candidate approaches and the reference timing. Let the KB rank them before you commit.

3. **Plan** — Call `build_plan` with your chosen approach and exact dimensions to get concrete thread block sizes, shared memory layout, and reduction strategy.

4. **Write** — Implement the kernel using the plan from step 3.

5. **Review** — Call `review_plan` with your kernel code before running evaluation. Fix any issues it identifies (wrong indexing, missing sync barriers, race conditions).

6. **On failure** — If evaluation fails (compile error or incorrect results), call `diagnose_failure` with the exact error message and your kernel code. If numerical correctness fails, also call `verify_code_math` with the tensor shapes and reduction dimensions.

### Tool Quick Reference

| Tool | Purpose |
|------|---------|
| `search_knowledge` | Find relevant kernel implementations and optimization patterns |
| `propose_hypothesis` | Rank candidate optimization approaches |
| `build_plan` | Get concrete implementation plan (block dims, shared mem, reductions) |
| `review_plan` | Review kernel code for correctness bugs before testing |
| `verify_code_math` | Check numerical correctness (softmax stability, normalization, etc.) |
| `diagnose_failure` | Diagnose compilation errors or incorrect results |
| `get_page` | Retrieve full KB page when a tool response cites a `[PageID]` |

### Search Tips

Include problem-specific context in every `search_knowledge` call:

- Exact tensor shapes (e.g., `"(112, 64, 512, 512)"`)
- Reduction dimension and size (e.g., `"reducing over C=64"`)
- Whether the reference uses an optimized library (e.g., `"reference uses F.scaled_dot_product_attention"`)
- Target GPU architecture (e.g., `"NVIDIA L4, compute capability 8.9, Ada Lovelace"`)

You must call `search_knowledge` at least once per problem and `propose_hypothesis` or `build_plan` at least once per problem. Combine KB recommendations with your own analysis — but do not skip the KB lookup.
