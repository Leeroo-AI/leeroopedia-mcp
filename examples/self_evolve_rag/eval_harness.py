"""
Frozen Evaluation Harness for Self-Evolving RAG Benchmark
=========================================================

DO NOT MODIFY this file. Both baseline and KB-assisted agents must use
these functions for all retrieval metric computation.

Metrics:
  - precision_at_k: fraction of top-K retrieved chunks that are substrings
    of any ground-truth document.
  - recall_at_k: fraction of ground-truth documents that have at least one
    matching chunk in the top-K retrieved results.

A retrieved chunk is considered "relevant" if its text appears as a
substring of any ground-truth document text. Since chunks are carved
from documents, this is the natural relevance definition.
"""

import json
from pathlib import Path


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_eval_queries(path: str = "eval_queries.json") -> dict:
    """
    Load the pre-built evaluation queries.

    Returns a dict with keys:
      - "evolution_queries": list of query dicts (train split)
      - "held_out_queries":  list of query dicts (test split)

    Each query dict has:
      - "query_id": str
      - "question": str
      - "ground_truth_doc_texts": list[str]  (full document texts)
      - "reference_answer": str
    """
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── Retrieval Metrics ────────────────────────────────────────────────────────

def precision_at_k(
    retrieved_chunk_texts: list[str],
    ground_truth_doc_texts: list[str],
    k: int = 5,
) -> float:
    """
    Precision@K — fraction of top-K retrieved chunks that are relevant.

    A chunk is "relevant" if its full text is a substring of any
    ground-truth document text.

    Args:
        retrieved_chunk_texts: texts of retrieved chunks, ordered by rank.
        ground_truth_doc_texts: full texts of ground-truth documents.
        k: number of top results to consider.

    Returns:
        Float in [0.0, 1.0].
    """
    top_k = retrieved_chunk_texts[:k]
    if not top_k:
        return 0.0

    relevant = 0
    for chunk in top_k:
        for gt_doc in ground_truth_doc_texts:
            if chunk in gt_doc:
                relevant += 1
                break

    return relevant / k


def recall_at_k(
    retrieved_chunk_texts: list[str],
    ground_truth_doc_texts: list[str],
    k: int = 5,
) -> float:
    """
    Recall@K — fraction of ground-truth documents covered by top-K chunks.

    A ground-truth document is "covered" if at least one chunk in the
    top-K is a substring of that document.

    Args:
        retrieved_chunk_texts: texts of retrieved chunks, ordered by rank.
        ground_truth_doc_texts: full texts of ground-truth documents.
        k: number of top results to consider.

    Returns:
        Float in [0.0, 1.0].
    """
    top_k = retrieved_chunk_texts[:k]
    if not ground_truth_doc_texts:
        return 0.0

    found = 0
    for gt_doc in ground_truth_doc_texts:
        for chunk in top_k:
            if chunk in gt_doc:
                found += 1
                break

    return found / len(ground_truth_doc_texts)


# ── Full Evaluation Orchestrator ─────────────────────────────────────────────

def evaluate_queries(
    queries: list[dict],
    retrieve_fn,
    k: int = 5,
) -> dict:
    """
    Run evaluation over a list of queries using the provided retrieval function.

    Args:
        queries: list of query dicts from eval_queries.json, each containing
                 "question" and "ground_truth_doc_texts".
        retrieve_fn: callable(question: str) -> list[str]
                     Must return a list of retrieved chunk TEXTS, ordered by rank.
        k: top-K for precision and recall.

    Returns:
        Dict with:
          - "retrieval_precision_at_k": float (averaged across queries)
          - "retrieval_recall_at_k": float (averaged across queries)
          - "num_queries": int
          - "per_query": list of per-query result dicts
    """
    precisions = []
    recalls = []
    per_query = []

    for q in queries:
        question = q["question"]
        gt_docs = q["ground_truth_doc_texts"]

        # Retrieve chunk texts via the caller's retrieval function
        chunk_texts = retrieve_fn(question)

        p = precision_at_k(chunk_texts, gt_docs, k=k)
        r = recall_at_k(chunk_texts, gt_docs, k=k)

        precisions.append(p)
        recalls.append(r)
        per_query.append({
            "query_id": q.get("query_id", ""),
            "question": question,
            "precision": p,
            "recall": r,
        })

    n = len(queries)
    avg_p = sum(precisions) / n if n else 0.0
    avg_r = sum(recalls) / n if n else 0.0

    return {
        "retrieval_precision_at_5": round(avg_p, 4),
        "retrieval_recall_at_5": round(avg_r, 4),
        "num_queries": n,
        "per_query": per_query,
    }
