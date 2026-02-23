# Leeroopedia Knowledge Base Tools — Reference

This document describes how to use the Leeroopedia MCP tools for the Self-Evolving RAG task. It is kept as a reference and is NOT included in the agent prompt.

---

## Leeroopedia Knowledge Base Tools

You have access to the Leeroopedia MCP tools. Use them to learn implementation patterns, but ALWAYS validate KB recommendations against your actual empirical results. The KB provides general best practices — your specific corpus (IBM TechQA) may behave differently.

**When to use KB tools:**
- At the START of each phase, to learn implementation patterns (chunking strategies, atomic swap patterns, hybrid retrieval fusion methods).
- When you encounter a BUG or ERROR you cannot resolve from code alone.
- When you need to verify correctness of a specific algorithm (e.g., score normalization formula).

**When NOT to use KB tools:**
- Do NOT use KB recommendations to override empirical results. If your data shows BM25 outperforms vector search, trust your data — even if the KB suggests semantic search is generally superior.
- Do NOT use `query_hyperparameter_priors` to set initial fusion weights or chunk sizes. Use the defaults specified in this proposal (0.5/0.5 weights, 512-token chunks) and let the evolution loop adapt based on actual metrics.
- Do NOT use KB recommendations to decide how many documents to re-chunk. The diagnosis logic should determine scope based on failure analysis, not KB heuristics.

**Tool-specific guidance:**

- **`search_knowledge`**: Use to look up implementation patterns (atomic index swaps, hybrid retrieval fusion, chunking strategies). Focus on HOW to implement correctly, not WHAT hyperparameters to use.

- **`build_plan`**: Use ONCE at the start to get a high-level implementation skeleton. Do not request plans for individual evolution rounds — the evolution logic should be data-driven, not plan-driven.

- **`verify_code_math`**: Use to verify correctness of score normalization, metric computation, and atomic swap logic. This is the highest-value tool — use it after writing critical code sections.

- **`diagnose_failure`**: Use when something goes WRONG (evaluation scores are zero, index swap causes errors, pipeline crashes). Do not use for design decisions.

- **`get_page`**: Use when any tool response cites a `[PageID]` and you need deeper detail.

- **`review_plan`**, **`propose_hypothesis`**, **`query_hyperparameter_priors`**: Use sparingly. These tools provide general ML wisdom that may not apply to this specific benchmark. Prefer empirical evidence from your own evaluation runs over generic KB recommendations.
