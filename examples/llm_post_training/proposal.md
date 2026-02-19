# Demo Proposal: LLM Post-Training & Deployment Pipeline

## Overview

Build a complete post-training and deployment pipeline for `Qwen/Qwen2.5-1.5B`. Starting from the base pretrained model, implement three tasks that produce a fine-tuned, merged, and efficiently served model.

All code must be runnable end-to-end on a single node with 8 GPUs, with no manual intervention between stages. 

---

## Task 1 — SFT + DPO Alignment with LoRA

Implement a two-stage alignment pipeline for `Qwen/Qwen2.5-1.5B`.

**Stage 1 — Supervised Fine-Tuning (SFT):**
Fine-tune the base model on 50k multi-turn conversations from `HuggingFaceH4/ultrachat_200k` (train_sft split).

**Stage 2 — Direct Preference Optimization (DPO):**
Continue training on preference data from `argilla/ultrafeedback-binarized-preferences-cleaned` (~60k general preference pairs, a cleaned version of UltraFeedback with TruthfulQA contamination removed).

---

## Task 2 — LoRA Merge

Take the LoRA adapter from Task 1 and produce a deployment-ready merged model.

**Merge:** Bake the LoRA adapter weights into the base model checkpoint.

**Verify:** Run a forward pass on the merged model with a fixed input and confirm outputs are consistent with the LoRA model.

Output: a full merged model saved to `./merged_model/`.

---

## Task 3 — Efficient Serving & Throughput Benchmark

Deploy the merged model from Task 2 with vLLM and benchmark throughput.

**Serve:** Launch the merged model with vLLM. Expose `/v1/chat/completions` supporting both standard and streaming (SSE) responses.

**Benchmark:** Measure decode throughput (tokens/second) at batch sizes 1 and 4 using a standardized prompt set. Report prefill and decode throughput separately. You only need to benchmark the trained merged model — no need to benchmark the base or instruct models for throughput.

---

## Task 4 — Quality Evaluation

Evaluate the trained model using the pre-built helper script `evaluate_model.py` (already provided in your workspace). This script wraps `lm-evaluation-harness` and runs the correct benchmarks. You only need to evaluate the trained merged model — no need to evaluate the base or instruct models.

**Step-by-step evaluation procedure:**

1. **Install lm-evaluation-harness** (if not already installed):
   ```bash
   pip install lm-eval
   ```

2. **Evaluate the trained (merged) model:**
   ```bash
   python evaluate_model.py --model ./merged_model --label trained
   ```

The script runs the **IFEval** benchmark and saves results to `./eval_results/`. It prints a summary at the end.

**Benchmark being run:**

- **IFEval (strict-prompt):** 541 prompts with verifiable instruction-following constraints (e.g. "write exactly 3 paragraphs", "include the word 'ocean' at least twice"). Scoring is binary pass/fail per prompt. No LLM judge needed. Report the **strict-prompt accuracy** metric. The task name in lm-eval-harness is `ifeval`.

**IMPORTANT:** Do NOT substitute different benchmarks. Use the exact task name `ifeval` as specified. The `evaluate_model.py` script handles this correctly — just run it as shown above.

---

## Final Report

After all evaluations complete, print a summary with:

| Model | IFEval (strict-prompt) |
|-------|------------------------|
| Trained (SFT+DPO) | X.XX% |

Plus throughput results from Task 3.

---

## Completion Requirement

Do NOT stop until the full pipeline has finished and you have reported the complete benchmark results (IFEval score) and throughput numbers (prefill and decode tokens/second at batch sizes 1 and 4) for the trained model. The pipeline must run end-to-end without interruption: training, merging, serving, evaluation, and benchmarking. Only consider the task done when final benchmark and throughput numbers are printed.

---

## Environment

You are running inside the `demo` conda environment. You are free to install any Python packages you need using `pip install` or `conda install`. Install whatever is required to complete the pipeline (e.g. TRL, PEFT, vLLM, lm-evaluation-harness, datasets, accelerate, etc.).

---

## Long-Running Commands

Training and evaluation steps may take minutes to hours. For any command that takes more than 2 minutes:

1. **Run in background**: `python script.py > script.log 2>&1 &`
2. **Monitor every 60 seconds**: `tail -20 script.log` to check progress
3. **Verify process is alive**: `ps aux | grep script.py`
4. **React to problems**: If no new output for 5+ minutes, check for errors, OOM kills, or hangs — then fix and retry

Never run a long command synchronously and wait blindly. Always actively monitor log output so you can detect and fix issues promptly.
