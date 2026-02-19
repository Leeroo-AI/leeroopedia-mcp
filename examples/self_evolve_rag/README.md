# Self-Evolving RAG Benchmark

This benchmark measures how a curated knowledge base (Leeroopedia MCP) improves an AI coding agent's ability to build a self-evolving RAG system from a specification.

Both agents receive the **same task**: build a FastAPI service over the RAGBench `techqa` dataset (IBM technical support notes) that ingests a corpus, answers questions via hybrid retrieval-augmented generation, and **automatically improves itself** over multiple evolution rounds by diagnosing retrieval failures, re-chunking documents, and adapting query strategies. The only difference is that one agent has access to Leeroopedia's knowledge base via MCP tools.

## Results

![Benchmark Results](analysis.png)

### Why +Leeroopedia performed better

1. **Atomic index swap pattern**: The Leeroopedia taught a blue-green collection pattern for ChromaDB (based on SQLMesh's virtual layer indirection and Lance's two-phase commit). The baseline used a simpler `threading.RLock` with in-memory pointer swap. The Agent with MCP's approach (`chunks_v0`, `chunks_v1`, etc.) with old-collection cleanup was more robust and is the pattern used by production vector databases.

2. **Hybrid score normalization**: The Leeroopedia taught DBSF (3-sigma) normalization for fusing BM25 and vector scores, explaining why raw scores are incomparable and comparing four normalization methods. The baseline used Reciprocal Rank Fusion, which works but loses score magnitude information. The Leeroopedia's DBSF approach let the evolution loop meaningfully adjust fusion weights based on empirical performance, enabling the shift from 0.50/0.50 to 0.40/0.60 (BM25-heavy) that drove the held-out improvement.

Round 1 produced the largest improvement (+1.5-1.8pp on held-out metrics). Round 2 continued improving the evolution set but saw a slight held-out pullback (within the 2% regression guard), consistent with diminishing returns. Incremental re-embedding worked correctly: round 1 reused 10,144 embeddings and only created 123 new ones.

---

## How to replicate

Full logs and outputs from our runs are available [here](https://drive.google.com/file/d/12eG6SYXGl3tuAIL2WpYTRElH1Otj_6pO/view?usp=sharing).

### Prerequisites

- Python 3.10+
- [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code) installed and configured
- An Anthropic API key **or** AWS Bedrock access
- A Leeroopedia API key
- An OpenAI API key (used by the agents to call GPT for embeddings and answer generation)

### 1. Set up environment variables

Create a `.env` file in this directory:

```
OPENAI_API_KEY=sk-...
LEEROOPEDIA_API_KEY="kpsk_..."
ANTHROPIC_API_KEY="sk-ant-..."

# Only needed if using --bedrock mode
# AWS_BEARER_TOKEN_BEDROCK="..."
# CLAUDE_CODE_USE_BEDROCK=1
# AWS_REGION=us-east-1
```

### 2. Set up the MCP config

Create `leeroopedia_mcp_config.json` in this directory:

```json
{
  "mcpServers": {
    "leeroopedia": {
      "command": "npx",
      "args": ["-y", "@anthropic/leeroopedia-mcp"],
      "env": {
        "LEEROOPEDIA_API_KEY": "<your key>"
      }
    }
  }
}
```

### 3. Install Python dependencies

```bash
pip install python-dotenv datasets
```

The agents install additional packages (`fastapi`, `uvicorn`, `chromadb`, `sentence-transformers`, `rank-bm25`, `openai`, etc.) inside their sandboxes during the run.

### 4. Run the benchmark

```bash
# Default: uses Anthropic API key
python run_benchmark.py
```

This runs the full pipeline:

1. **Eval data prep** -- Generates `eval_queries.json` from the RAGBench `techqa` dataset (~1,800 queries split into evolution and held-out sets).
2. **Phase 1 (With Leeroopedia)** -- Runs Claude Code with `proposal.md` as the task, plus Leeroopedia MCP tools. The agent builds the RAG service, runs ingestion, baseline evaluation, and 2+ evolution rounds.
3. **Phase 2 (Baseline)** -- Runs Claude Code with the same `proposal.md` but no MCP tools.

Both agents work in isolated `/tmp` sandboxes. Generated code is copied to `workspaces/task2_self_evolving_rag/with_kb/` and `workspaces/task2_self_evolving_rag/baseline/` after each phase.

> **Tip:** For better Leeroopedia usage, append the contents of [`leeroopedia_tools.md`](leeroopedia_tools.md) to the end of `proposal.md`. This teaches the agent when and how to call the KB tools.