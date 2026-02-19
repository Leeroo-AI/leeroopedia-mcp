# Demo Task 2 Proposal: Self-Evolving RAG System

## Overview

Build a RAG system that **monitors its own retrieval and answer quality, identifies weaknesses, and automatically improves itself** — without human intervention.

The system starts with a naive baseline (fixed-size chunking, single-vector retrieval). It then runs an evaluation loop over a held-out query set, diagnoses where retrieval is failing, and applies targeted fixes: re-chunking poorly-performing document regions, adjusting retrieval strategy, and rewriting weak queries. After each evolution round, it re-evaluates to confirm improvement.

The deliverable is a Python service with an HTTP API that supports: ingestion, querying, evaluation, and a self-evolution endpoint that triggers the improvement cycle.

---

## Environment

You are running inside the `kapso-bench` conda environment. You are free to install any Python packages you need using `pip install` or `conda install`. Install whatever is required to complete the pipeline (e.g., FastAPI, uvicorn, chromadb, lancedb, sentence-transformers, rank-bm25, ragas, datasets, openai, tiktoken, etc.).

---

## Task — Self-Evolving RAG with Feedback-Driven Improvement

Implement a FastAPI service that ingests a technical documentation corpus, answers questions via retrieval-augmented generation, and improves itself over multiple evolution rounds.

### Phase 1 — Baseline RAG

Build a working RAG pipeline over the RAGBench `techqa` dataset (IBM technical support notes).

**Ingestion:**
1. Load the `techqa` subset from HuggingFace (`rungalileo/ragbench`, config `techqa`).
2. Extract the unique source documents from the `documents` column.
3. Chunk documents using a naive fixed-size strategy (e.g., 512 tokens with 64-token overlap).
4. Embed chunks using an embedding model and store in a vector database.
5. Build a BM25 index over the same chunks for lexical retrieval.

**Querying:**
1. Accept a question via `POST /query`.
2. Retrieve top-K chunks using hybrid retrieval (vector similarity + BM25), with configurable fusion weights.
3. Pass retrieved chunks as context to an LLM and generate an answer.
4. Return: `answer`, `retrieved_chunks` (with chunk IDs and scores), `retrieval_method`, `latency_ms`.

**Evaluation:**
1. `POST /evaluate` accepts a `split` parameter: `"evolution"` or `"held_out"` (default: `"held_out"`).
2. Runs the specified query set through the pipeline.
3. For each query, compare retrieved chunks against the dataset's ground-truth supporting documents.
4. Compute and return: retrieval precision@5 and retrieval recall@5, using the provided `eval_harness.py` functions.
5. Store the evaluation results with a round number and split name for tracking improvement over time.

### Pre-built Evaluation Harness (MANDATORY)

Two files are provided in your working directory:

- **`eval_queries.json`** — Contains the pre-built evaluation queries for both the evolution set and held-out test set. Each query has `question`, `ground_truth_doc_texts` (list of full document texts), and `reference_answer`. You MUST load queries from this file for evaluation. Do NOT construct your own evaluation queries from the dataset.

- **`eval_harness.py`** — Contains the metric computation functions: `precision_at_k()`, `recall_at_k()`, and `evaluate_queries()`. You MUST use these functions for all retrieval metric computation. Do NOT implement your own metric functions.

**Metric definitions (implemented in eval_harness.py — do not reimplement):**
- **Precision@5**: A retrieved chunk is "relevant" if its full text is a substring of any ground-truth document text. P@5 = (relevant chunks in top-5) / 5.
- **Recall@5**: Fraction of ground-truth documents that have at least one matching chunk (by substring) in the top-5 retrieved results.

**How to use the harness in your `/evaluate` endpoint:**
1. `from eval_harness import load_eval_queries, precision_at_k, recall_at_k`
2. Load queries: `data = load_eval_queries("eval_queries.json")`
3. Select the split: `queries = data["evolution_queries"]` or `data["held_out_queries"]`
4. For each query, retrieve top-5 chunks and extract their **text** content
5. Call `precision_at_k(chunk_texts, q["ground_truth_doc_texts"])` and `recall_at_k(chunk_texts, q["ground_truth_doc_texts"])`
6. Average across all queries in the split

**You may also use `eval_harness.evaluate_queries(queries, retrieve_fn)`** as a convenience wrapper — pass it a `retrieve_fn(question) -> list[str]` that returns chunk texts, and it returns averaged metrics plus per-query details.

### Phase 2 — Self-Evolution Loop

**Critical: the evolution set is for diagnosis ONLY.** Improving evolution-set metrics while degrading held-out metrics is overfitting — not improvement. A good evolution strategy makes small, targeted changes (re-chunk a few dozen documents, adjust weights slightly) and verifies held-out generalization at each step. Aggressive changes (re-chunking hundreds of documents, large weight swings) risk overfitting to the evolution set.

Implement `POST /evolve` that triggers one improvement cycle:

**Step 1 — Diagnose:**
- Run `/evaluate` on the **evolution set only** (`split="evolution"`).
- Identify the bottom 20% of queries by retrieval recall (the queries where the system failed to find the right documents).
- For each failing query, analyze WHY retrieval failed: was the relevant content split across chunk boundaries? Was the query too different from the document vocabulary? Was the relevant chunk ranked low but present?
- The held-out test set must NOT be used for diagnosis or adaptation.

**Step 2 — Adapt chunking:**
- ONLY re-chunk documents where retrieval completely failed — i.e., no relevant chunk appeared even in an extended top-20 retrieval. Do NOT re-chunk documents where the relevant chunk exists but was merely ranked low (that is a ranking problem for Step 3, not a chunking problem). Expect to re-chunk roughly 20–80 documents per round (1–3% of corpus), NOT hundreds. Over-aggressive re-chunking destabilizes the index and causes overfitting.
- For documents that genuinely need re-chunking, use a different strategy based on the failure mode:
  - If chunks are too large (relevant content diluted): split into smaller chunks.
  - If relevant content was split across boundaries: use overlapping sentence-window chunking.
- Re-embed only the affected chunks (not the entire corpus).
- Swap the updated chunks into the index atomically — queries during the swap must not see a partial index.

**Step 3 — Adapt retrieval:**
- For queries where the relevant chunk existed but was ranked too low:
  - Generate a query expansion (add synonyms/related terms from the document vocabulary).
  - Store the expansion as a learned rewrite rule for similar future queries.
- Adjust the hybrid fusion weights based on which retrieval method (vector vs. BM25) performed better on the failing queries.

**Step 4 — Re-evaluate:**
- Run evaluation on **both** the evolution set and the held-out test set.
- Store results as the next round.
- Return a comparison: metrics at round N-1 vs. round N, with per-query deltas.
- The held-out test set metrics are the ones that matter — they show whether improvements generalize beyond the queries used for diagnosis.
- **Regression guard**: If any held-out metric degrades by more than 2% from the previous round, log a warning. The goal is monotonic improvement on the held-out set. Improving evolution-set metrics while degrading held-out metrics means the adaptation was too aggressive — tighten the scope for the next round.

### Phase 3 — Evolution History

Implement `GET /evolution/history` that returns the full improvement trajectory:

```json
{
  "rounds": [
    {
      "round": 0,
      "timestamp": "2026-02-13T10:00:00Z",
      "evolution_set_metrics": {
        "retrieval_precision_at_5": 0.42,
        "retrieval_recall_at_5": 0.38
      },
      "held_out_metrics": {
        "retrieval_precision_at_5": 0.40,
        "retrieval_recall_at_5": 0.36
      },
      "changes_applied": "baseline — naive 512-token chunking, equal fusion weights"
    },
    {
      "round": 1,
      "timestamp": "2026-02-13T10:05:00Z",
      "evolution_set_metrics": {
        "retrieval_precision_at_5": 0.55,
        "retrieval_recall_at_5": 0.50
      },
      "held_out_metrics": {
        "retrieval_precision_at_5": 0.48,
        "retrieval_recall_at_5": 0.44
      },
      "changes_applied": "re-chunked 23 documents, adjusted fusion weights to 0.6 vector / 0.4 BM25"
    }
  ]
}
```

### Core Technical Requirements

**Atomic index updates:**
When re-chunking triggers a re-index, write to a new snapshot and swap via an alias (or equivalent). Live queries must never see a half-built index. Do NOT update chunks in-place while queries are running.

**Incremental re-embedding:**
Only re-embed chunks that changed. Track document content hashes to skip unchanged content. Do NOT re-embed the entire corpus on every evolution round. When re-chunking produces new chunk IDs, you must still reuse embeddings for unchanged chunks from non-affected documents — only the re-chunked documents' chunks should be newly embedded. Log the count of reused vs. newly embedded chunks each round. If the reused count is 0 for a non-trivial run, you have a bug — fix it before proceeding.

**Evaluation correctness:**
You MUST use the provided `eval_harness.py` for all metric computation and `eval_queries.json` for all evaluation queries. Do NOT implement your own precision/recall functions or construct your own query sets from the dataset for evaluation. The harness ensures identical metrics across all benchmark runs.

**Deterministic rounds:**
Given the same corpus and query set, running `/evolve` twice from the same state must produce the same diagnosis and the same changes. Use fixed random seeds where applicable.

**Recommended models:**
Use OpenAI `gpt-5.2` for answer generation. Use OpenAI `text-embedding-3-small` for embeddings. Using the same models ensures fair comparison between baseline and KB-assisted runs.

**Initial fusion weights:**
Start with equal fusion weights (0.5 vector / 0.5 BM25). Do NOT start with a strong bias toward either method without evidence. Let the evolution loop adjust weights based on empirical data from the diagnosis step. The optimal balance depends on corpus characteristics, which you will discover through the evolution loop.

---

## Test Data

The system is evaluated using the **RAGBench `techqa` subset** from HuggingFace (`rungalileo/ragbench`).

### Dataset Details

- **Source**: IBM TechQA — real questions from IBM developer forums, answered against IBM Technotes (technical support documentation).
- **HuggingFace path**: `rungalileo/ragbench` with config name `techqa`.
- **Size**: ~1,800 examples across splits.
- **Columns**: `question`, `documents` (list of retrieved document texts), `response` (reference answer), sentence-level support annotations.

### Data Splits

The dataset must be split into three parts:

| Split | Source | Purpose | Used by `/evolve`? |
|---|---|---|---|
| **Corpus** | All unique documents from `documents` column across all splits | Ingested and indexed | — |
| **Evolution set** | `train` split queries | Diagnosis and adaptation — the evolution loop analyzes failures on this set to decide what to re-chunk and how to adapt retrieval | YES |
| **Held-out test set** | **Complete** `test` split queries (ALL examples, no subsetting) | Measures whether improvements generalize — never seen during evolution | NO |

### Corpus Construction (deterministic)

1. Load the dataset: `load_dataset("rungalileo/ragbench", "techqa")`.
2. Extract unique documents from the `documents` column across all splits — these form the ingestion corpus.
3. Use the `train` split as the **evolution set** (for `/evolve` diagnosis).
4. Use the **complete** `test` split as the **held-out test set** (for generalization measurement). You must use every example in the test split — do NOT sample, truncate, or use a subset. The full test split is the held-out evaluation set.
5. Each example provides: `question`, ground-truth `documents` (the relevant documents), and `response` (reference answer).

**Critical rule:** The evolution loop (`/evolve`) must ONLY use the evolution set for diagnosis and adaptation decisions. It must NEVER read, analyze, or optimize for queries in the held-out test set. The held-out test set is evaluated after each round solely to measure generalization.

**Critical rule:** The held-out test set must contain ALL examples from the `test` split — no subsampling, no truncation, no random subset. Evaluation metrics must reflect performance across the entire test split.

### Rules

- The agent must build the complete system (baseline + evolution loop) first, then demonstrate it on the test data.
- The agent must NOT hard-code fixes for specific test queries. The evolution mechanism must be general-purpose.
- The evolution loop must be triggered via the API endpoint, not by manual code changes between rounds.

---

## Completion Requirement

Do NOT stop until the full pipeline has finished and you have reported the complete evolution results: baseline metrics on BOTH the evolution set and held-out test set (retrieval precision@5, recall@5) and the same metrics after each evolution round, alongside a summary of what changed and why. The pipeline must run end-to-end without interruption: corpus ingestion, baseline indexing, baseline evaluation on both sets, at least 2 evolution rounds (diagnose on evolution set, re-chunk, re-index, re-evaluate on both sets), and evolution history output. Only consider the task done when:

1. The service is running and accepting requests.
2. The techqa corpus is ingested and indexed.
3. Baseline evaluation (round 0) metrics are printed for both evolution set and held-out test set.
4. At least 2 evolution rounds have completed via `/evolve`.
5. The full evolution history is printed showing metrics on both sets at every round.
6. A before/after comparison for at least one query shows how retrieved chunks changed.
7. A final summary states whether each **held-out test set** metric improved, stayed flat, or degraded across rounds.
