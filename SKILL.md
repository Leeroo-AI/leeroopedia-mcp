---
name: leeroopedia-mcp
description: Use Leeroopedia MCP to fetch grounded ML/AI best practices, build and review ML plans, debug failures, verify code/math correctness, and expand KB citations via get_page.
---

# Leeroopedia MCP

Leeroopedia is an ML/AI knowledge wiki. This skill teaches you when and how to call the Leeroopedia MCP tools so answers are grounded in documented best practices (not guesswork).

## Default behavior

Use Leeroopedia MCP whenever the user asks anything that depends on **ML/AI framework specifics** or **best practices** (fine-tuning, post-training, inference serving, CUDA/Triton kernels, distributed training, RAG/agents, evaluation, config formats, API contracts, performance tuning).

If the question is purely general software engineering (no ML/AI-specific uncertainty), you may answer without tools.

## Grounding policy

1. **Prefer tool-grounded answers over memory.** If you are not 100% sure about an ML/AI detail, call a tool.
2. **Cite sources.** When tools return `[PageID]` citations, preserve them in your final answer.
3. **Expand citations when needed.** If the user asks “show me the source”, “give the full reference”, or you need precise details, call `get_page` on the cited `[PageID]`.
4. **Use parallel searches.** For ambiguous topics, call `search_knowledge` multiple times with different angles (faster and higher recall than a single broad query).

## Tool selection guide

### 1) `search_knowledge(query, context?)`
**Use when:** you need documented facts: framework behavior, APIs, configs, conventions, design patterns, tradeoffs.  
**How to query well:**
- Include framework + component + intent + constraints (model size, GPUs, latency/throughput target, memory limit, etc.)
- Ask narrow questions; do multiple calls instead of one giant query.
- Add `context` if you’re implementing a specific system.

**Good query patterns**
- “How does X work in Y (version)? What are the gotchas?”
- “What config fields are required for Z?”
- “Best practices for A given constraints B”

### 2) `build_plan(goal, constraints?)`
**Use when:** the user wants an end-to-end or multi-step implementation plan (pipelines, training runs, deployments, evaluations).  
**Output expectation:** overview, key specs, numbered steps, validation criteria.

### 3) `review_plan(proposal, goal)`
**Use when:** the user has a draft plan (or you wrote one) and wants a sanity check against best practices.  
**Output expectation:** approvals, risks, suggested improvements.

### 4) `verify_code_math(code_snippet, concept_name)`
**Use when:** verifying correctness of math/ML logic, algorithmic implementation, or API usage in a critical snippet.  
**Output expectation:** Pass/Fail + explanation of discrepancies.

### 5) `diagnose_failure(symptoms, logs)`
**Use when:** training/inference/deployment is failing (OOM, divergence, NaNs, hangs, bad throughput, wrong outputs, dependency conflicts).  
**Input quality:** include the most relevant error lines + minimal reproduction context.

### 6) `propose_hypothesis(current_status, recent_experiments?)`
**Use when:** the user is stuck or needs ranked next steps (design choices, debugging strategy, alternative approaches).  
**Output expectation:** ranked hypotheses + rationale + suggested experiments.

### 7) `query_hyperparameter_priors(query)`
**Use when:** the user asks “what LR / batch size / LoRA rank / weight decay / scheduler / etc should I use?”  
**Output expectation:** suggested values/ranges + justification.

### 8) `get_page(page_id)`
**Use when:** you need the full KB page for a cited source or you already know the exact page ID.  
**Output expectation:** full page content in markdown.

## Canonical workflows

### A) “How do I implement X?”
1. `build_plan(goal, constraints)`
2. `search_knowledge` for the 2–4 most uncertain steps (APIs, config, pitfalls)
3. Produce implementation guidance with `[PageID]` citations
4. Use `get_page` if the user asks for full source or details

### B) “Is my plan correct?”
1. `review_plan(proposal, goal)`
2. If a risk depends on a specific API/config detail, follow up with `search_knowledge`
3. Return an improved plan + validation checklist

### C) “My run crashed / performance is bad”
1. `diagnose_failure(symptoms, logs)`
2. If tuning is required, `query_hyperparameter_priors`
3. If multiple plausible causes, `propose_hypothesis` to rank next experiments

### D) “Is this code correct?”
1. `verify_code_math(code_snippet, concept_name)`
2. If the correct behavior depends on framework-specific contracts, `search_knowledge`
3. Provide corrected snippet + rationale

## Examples (copy/paste style)

### Example 1: framework behavior
- Call: `search_knowledge`
- Query: “How does vLLM handle tensor parallelism and kv-cache memory? Common throughput pitfalls?”
- Context: “Serving a 7B model on 2×A100, target >200 tok/s”

### Example 2: end-to-end post-training
- Call: `build_plan`
- Goal: “SFT + preference optimization + merge + vLLM deploy + IFEval for Qwen2.5-1.5B”
- Constraints: “8×A100, minimize wall time, reproducible evaluation”

### Example 3: review a plan
- Call: `review_plan`
- Proposal: “Load in 4-bit, apply LoRA rank 64 everywhere, train 3 epochs with lr=2e-5, merge and save”
- Goal: “QLoRA instruction tuning for a 8B model”

### Example 4: verify a critical snippet
- Call: `verify_code_math`
- Code Snippet: `lora_scaling = lora_alpha / lora_r`
- Concept Name: “LoRA scaling factor computation”

### Example 5: debug an OOM
- Call: `diagnose_failure`
- Symptoms: “OOM during QLoRA fine-tuning on A100 40GB, batch size 4, seq len 4096”
- Logs: “RuntimeError: CUDA out of memory. Tried to allocate ...”

### Example 6: pick next optimization steps
- Call: `propose_hypothesis`
- Current Status: “RAG on technical docs has low recall on multi-hop questions; naive top-k retrieval fails”
- Recent Experiments: “Tried larger chunk overlap; marginal gains”

### Example 7: hyperparameter defaults
- Call: `query_hyperparameter_priors`
- Query: “Learning rate, LoRA rank/alpha, batch size for QLoRA fine-tuning Llama-3 8B on A100 80GB”

### Example 8: expand a citation
- Call: `get_page`
- Page ID: “Workflow/QLoRA_Finetuning”

## Failure modes and what to do

- If results seem thin: rerun `search_knowledge` with 2–3 narrower queries.
- If the user needs strict correctness: favor `get_page` expansion and quote the relevant section (briefly) with the `[PageID]`.
- If a tool call fails (auth/credits/timeout): explain the failure and suggest the minimal next step (e.g., refine query, increase poll max wait, confirm API key).

## Output style

- Be direct and implementation-oriented.
- Prefer checklists, numbered steps, and validation criteria.
- Keep citations as `[PageID]` inline where they support key claims.