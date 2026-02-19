"""
Benchmark pipeline: generate, enrich, tag, and score coding tasks.

Combines four steps into one CLI:
  1. generate   - Create hard coding tasks per repo using GPT-5.2 web search
  2. enrich     - Add hardware requirements & credentials needed
  3. tag        - Assign domain tags (e.g. "llm-agents", "backend-api")
  4. score      - Rate how real-world each task feels (1-10)

Usage:
  python generate_benchmarks.py generate [--batch N] [filter_key ...]
  python generate_benchmarks.py enrich [filter_key ...]
  python generate_benchmarks.py tag
  python generate_benchmarks.py score
  python generate_benchmarks.py --list-keys
"""

import asyncio
import json
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI

load_dotenv()

# ── File paths ───────────────────────────────────────────────────────────────
INPUT_FILE = "../../repos.json"          # flat list of GitHub URLs
BENCHMARKS_FILE = "repo_benchmarks.json" # output / working file

# ── Concurrency defaults ────────────────────────────────────────────────────
DEFAULT_BATCH_SIZE = 5   # repos per batch for generate
ENRICH_BATCH_SIZE = 5    # tasks per API call for enrich
ENRICH_MAX_WORKERS = 3   # parallel API calls for enrich
ASYNC_MAX_CONCURRENCY = 10  # parallel calls for tag / score

# ── Model ────────────────────────────────────────────────────────────────────
MODEL = "gpt-5.2"


# ═══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════════════════

def strip_code_fences(text):
    """Remove markdown code fences wrapping JSON output."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:])
        if text.endswith("```"):
            text = text[:-3]
    return text.strip()


def url_to_dir_name(url):
    """
    Derive a dir_name key from a GitHub URL.
    e.g. "https://github.com/langchain-ai/langgraph" -> "langchain_ai_langgraph"

    Replaces hyphens and dots with underscores, lowercases everything.
    """
    url = url.rstrip("/")
    match = re.search(r"github\.com/([^/]+)/([^/]+)", url)
    if not match:
        raise ValueError(f"Cannot parse GitHub URL: {url}")
    org, repo = match.group(1), match.group(2)
    org = re.sub(r"[-.]", "_", org).lower()
    repo = re.sub(r"[-.]", "_", repo).lower()
    return f"{org}_{repo}"


def build_repo_dict(urls):
    """Convert a list of GitHub URLs into a dict keyed by dir_name."""
    result = {}
    for url in urls:
        dir_name = url_to_dir_name(url)
        result[dir_name] = {"url": url}
    return result


def load_benchmarks():
    """Load existing benchmarks from disk (or empty dict)."""
    if os.path.exists(BENCHMARKS_FILE):
        with open(BENCHMARKS_FILE) as f:
            return json.load(f)
    return {}


def save_benchmarks(benchmarks):
    """Save benchmarks to disk."""
    with open(BENCHMARKS_FILE, "w") as f:
        json.dump(benchmarks, f, indent=2, sort_keys=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Step 1: Generate tasks
# ═══════════════════════════════════════════════════════════════════════════════

def generate_tasks_for_url(client, url):
    """
    Use OpenAI GPT-5.2 with web search and high reasoning
    to generate 2 hard coding tasks for a given repo URL.
    Tasks sound like real-world developer questions, not hyper-specific internals.
    """
    prompt = f"""Research the GitHub repository {url} thoroughly using web search.

Then create TWO hard but realistic implementation tasks that a developer might 
give to a coding agent (like Cursor, Copilot, or Claude Code).

RULES for each task:
- The developer is asking the agent to BUILD or IMPLEMENT something — a feature, 
  a pipeline, an integration, a tool, a script, etc.
- It should sound like a real project goal: "I want to build...", "Implement a...", 
  "Set up a pipeline that...", "Create a service that..."
- Do NOT mention the repo name, GitHub org name, or specific internal class/file names
- Keep it practical and real-world — something a developer would actually want to build 
  using this technology
- It should be hard enough that a coding agent without deep knowledge of this project 
  would produce a naive or incorrect implementation
- But a coding agent WITH a knowledge base about this project would implement it correctly
- NOT a debugging question or a "why is X broken" question — it's a BUILD task

For each task, also provide 5-6 EVALUATION CRITERIA — specific requirements that a 
correct implementation must satisfy. Each criterion should be:
- A single sentence, verifiable as pass/fail by reviewing the code output
- Testing for details that require deep project knowledge to get right
- Not mentioning the repo name (stay generic like the task itself)
These criteria will be used to score and compare agent outputs.

Return ONLY a valid JSON object (no markdown, no code fences):
{{
  "tasks": [
    {{
      "task": "the implementation task (2-3 sentences, natural, NO repo/org names)",
      "difficulty_reason": "why a naive agent would struggle (1 sentence)",
      "knowledge_needed": "what repo knowledge helps solve this (for our records)",
      "evaluation_criteria": [
        "criterion 1 — a specific, pass/fail requirement",
        "criterion 2",
        "criterion 3",
        "criterion 4",
        "criterion 5"
      ]
    }},
    {{
      "task": "second implementation task (2-3 sentences, natural, NO repo/org names)",
      "difficulty_reason": "why a naive agent would struggle (1 sentence)",
      "knowledge_needed": "what repo knowledge helps solve this (for our records)",
      "evaluation_criteria": [
        "criterion 1 — a specific, pass/fail requirement",
        "criterion 2",
        "criterion 3",
        "criterion 4",
        "criterion 5"
      ]
    }}
  ]
}}"""

    response = client.responses.create(
        model=MODEL,
        tools=[{"type": "web_search"}],
        reasoning={"effort": "high"},
        input=prompt,
    )

    text = strip_code_fences(response.output_text)
    return json.loads(text)


def cmd_generate(filter_keys=None, batch_size=DEFAULT_BATCH_SIZE):
    """
    Generate benchmarks for repos listed in repos.json.

    filter_keys: optional list of dir_name keys to process (for testing).
    batch_size:  how many repos to process concurrently per batch.
    """
    # Load repo URLs from the flat list
    with open(INPUT_FILE) as f:
        urls = json.load(f)

    # Convert flat URL list to a dict keyed by dir_name
    repos = build_repo_dict(urls)

    # Warn about unrecognised filter keys
    if filter_keys:
        missing = [k for k in filter_keys if k not in repos]
        if missing:
            print(f"WARNING: filter keys not found in repos.json: {missing}")
            print(f"Available keys: {sorted(repos.keys())}")

    # Load existing benchmarks for resuming
    benchmarks = load_benchmarks()

    # Determine which repos to process
    if filter_keys:
        to_process = {k: repos[k] for k in filter_keys if k in repos}
    else:
        # Skip repos that already have benchmarks
        to_process = {k: v for k, v in repos.items() if k not in benchmarks}

    total = len(to_process)
    print(f"Repos to process: {total}")
    print(f"Batch size: {batch_size}")
    if not filter_keys:
        print(f"Already done: {len(benchmarks)}")

    client = OpenAI()

    # Split into batches and process
    items = list(to_process.items())
    num_batches = (total + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, total)
        batch = items[start:end]

        print(f"\n--- Batch {batch_idx+1}/{num_batches} "
              f"(repos {start+1}-{end} of {total}) ---")

        # Fire off all API calls in this batch concurrently
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            future_to_repo = {}
            for dir_name, repo_info in batch:
                url = repo_info["url"]
                print(f"  Starting: {dir_name} ({url})")
                future = executor.submit(generate_tasks_for_url, client, url)
                future_to_repo[future] = (dir_name, url)

            for future in as_completed(future_to_repo):
                dir_name, url = future_to_repo[future]
                try:
                    result = future.result()
                    benchmarks[dir_name] = {
                        "url": url,
                        "tasks": result["tasks"],
                    }
                    print(f"  Done: {dir_name}")
                    for i, t in enumerate(result["tasks"]):
                        print(f"    Task {i+1}: {t['task'][:100]}...")
                except Exception as e:
                    print(f"  ERROR [{dir_name}]: {e}")

        # Save progress after each batch
        save_benchmarks(benchmarks)
        print(f"  Saved. Total benchmarks so far: {len(benchmarks)}")

        # Small delay between batches to be nice to the API
        if batch_idx < num_batches - 1:
            time.sleep(2)

    print(f"\nDone! {len(benchmarks)} benchmarks total.")


# ═══════════════════════════════════════════════════════════════════════════════
# Step 2: Enrich (hardware requirements + credentials)
# ═══════════════════════════════════════════════════════════════════════════════

def enrich_batch(client, batch_items):
    """
    Enrich a batch of tasks in a single API call.
    batch_items: list of (dir_name, task_index, task_text, repo_url, repo_description)
    Returns: list of (dir_name, task_index, result_dict)
    """
    # Build the prompt with numbered tasks
    task_list = ""
    for i, (dir_name, task_idx, task_text, repo_url, repo_desc) in enumerate(batch_items):
        task_list += f"\n--- Task {i+1} ---\n"
        task_list += f"Project: {repo_url}\n"
        task_list += f"Project description: {repo_desc}\n"
        task_list += f"Task: {task_text}\n"

    prompt = f"""For each of the following implementation tasks, determine:
1. What HARDWARE is required to run/develop this task (GPU type/VRAM, RAM, disk, etc.)
2. What CREDENTIALS or API keys the developer needs (e.g., HuggingFace token, AWS keys, etc.)

Be practical and specific. If no special hardware is needed, say so.
If no credentials are needed, say so.
{task_list}

Return ONLY a valid JSON object (no markdown, no code fences) mapping task numbers
to their requirements:
{{
  "1": {{
    "hardware_requirements": "specific hardware needed (1-2 sentences)",
    "credentials_needed": "specific credentials/API keys/tokens needed (1-2 sentences)"
  }},
  "2": {{
    "hardware_requirements": "...",
    "credentials_needed": "..."
  }}
}}"""

    response = client.responses.create(
        model=MODEL,
        tools=[{"type": "web_search"}],
        reasoning={"effort": "medium"},
        input=prompt,
    )

    text = strip_code_fences(response.output_text)
    parsed = json.loads(text)

    # Map results back to (dir_name, task_index, result)
    results = []
    for i, item in enumerate(batch_items):
        dir_name, task_idx = item[0], item[1]
        key = str(i + 1)
        if key in parsed:
            results.append((dir_name, task_idx, parsed[key]))
        else:
            print(f"    WARNING: No result for task {key} ({dir_name})")
    return results


def cmd_enrich(filter_keys=None):
    """
    Enrich benchmark tasks with hardware and credentials info.
    filter_keys: optional list of dir_name keys to process (for testing).
    """
    benchmarks = load_benchmarks()
    client = OpenAI()

    # Collect all tasks that need enrichment
    pending = []
    for dir_name, repo_data in benchmarks.items():
        if filter_keys and dir_name not in filter_keys:
            continue
        url = repo_data["url"]
        description = repo_data.get("description", "")
        for i, task in enumerate(repo_data.get("tasks", [])):
            # Skip already enriched
            if "hardware_requirements" in task:
                continue
            pending.append((dir_name, i, task["task"], url, description))

    print(f"Tasks to enrich: {len(pending)}")
    if not pending:
        print("Nothing to do.")
        return

    # Split into batches
    batches = [pending[i:i + ENRICH_BATCH_SIZE]
               for i in range(0, len(pending), ENRICH_BATCH_SIZE)]
    print(f"Batches: {len(batches)} (size={ENRICH_BATCH_SIZE}, workers={ENRICH_MAX_WORKERS})")

    # Process batches in parallel
    completed = 0
    with ThreadPoolExecutor(max_workers=ENRICH_MAX_WORKERS) as executor:
        futures = {}
        for batch_idx, batch in enumerate(batches):
            future = executor.submit(enrich_batch, client, batch)
            futures[future] = batch_idx

        for future in as_completed(futures):
            batch_idx = futures[future]
            try:
                results = future.result()
                for dir_name, task_idx, result in results:
                    task = benchmarks[dir_name]["tasks"][task_idx]
                    task["hardware_requirements"] = result["hardware_requirements"]
                    task["credentials_needed"] = result["credentials_needed"]
                    completed += 1

                names = [r[0] for r in results]
                print(f"  Batch {batch_idx+1}/{len(batches)} done "
                      f"({completed}/{len(pending)} tasks): {names}")

            except Exception as e:
                batch = batches[batch_idx]
                names = [item[0] for item in batch]
                print(f"  Batch {batch_idx+1}/{len(batches)} ERROR: {e} ({names})")

            # Save progress after each batch completes
            save_benchmarks(benchmarks)

    print(f"\nDone! Enriched {completed}/{len(pending)} tasks.")


# ═══════════════════════════════════════════════════════════════════════════════
# Step 3: Tag domain labels
# ═══════════════════════════════════════════════════════════════════════════════

DOMAIN_TAG_SYSTEM_PROMPT = """You are an expert software engineering categorizer.
You will be given a coding task description and the repository it belongs to.

Your job is to assign **1 to 3 broad domain tags** that describe what area of
software engineering this task falls into.

Pick tags from this list (use these exact strings). You may pick 1-3:

General software engineering domains:
- "web-development"
- "backend-api"
- "frontend-ui"
- "mobile-development"
- "data-engineering"
- "devops-infra"
- "cli-tooling"
- "database"
- "networking"
- "security"
- "testing"
- "cloud-services"
- "distributed-systems"
- "embedded-iot"
- "game-development"
- "scientific-computing"
- "developer-tools"
- "automation"
- "file-systems-storage"
- "observability-monitoring"
- "blockchain-crypto"
- "robotics"

ML / AI specific domains (use these instead of generic "machine-learning"):
- "llm-inference"           (LLM serving, inference optimization, quantization, vLLM, TensorRT-LLM)
- "llm-training"            (pre-training, fine-tuning, RLHF, LoRA, distributed training of LLMs)
- "llm-agents"              (agentic workflows, tool use, chains, orchestration frameworks)
- "rag-search"              (retrieval-augmented generation, vector search, embeddings, knowledge bases)
- "computer-vision"         (image classification, detection, segmentation, OCR)
- "image-generation"        (diffusion models, GANs, image synthesis, style transfer)
- "video-multimedia"        (video generation, processing, editing, streaming)
- "audio-speech"            (TTS, ASR, speech processing, audio generation)
- "nlp-text"                (text classification, NER, summarization, translation — non-LLM)
- "recommender-systems"     (recommendation engines, collaborative filtering, ranking)
- "ml-ops"                  (experiment tracking, model registry, pipeline orchestration, deployment)
- "ml-frameworks"           (model format conversion, runtime optimization, training frameworks)
- "reinforcement-learning"  (RL algorithms, environments, policy optimization)
- "tabular-ml"              (classical ML on structured data — XGBoost, sklearn, feature engineering)

If none of the above fit well, you may create ONE custom tag in lowercase-kebab-case.

Respond with ONLY valid JSON (no markdown fences):
{
  "domain_tags": ["tag1", "tag2"]
}
"""


def _build_tag_user_prompt(repo_key, repo_desc, task_text):
    """Build the user prompt for domain tagging."""
    return (
        f"Repository: {repo_key}\n"
        f"Repository description: {repo_desc}\n\n"
        f"Task:\n{task_text}"
    )


async def _tag_single_task(client, semaphore, repo_key, repo_desc, task_text, task_idx):
    """Call GPT-5.2 to tag a single task. Returns parsed JSON dict."""
    async with semaphore:
        user_msg = _build_tag_user_prompt(repo_key, repo_desc, task_text)
        try:
            resp = await client.chat.completions.create(
                model=MODEL,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": DOMAIN_TAG_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
            )
            raw = strip_code_fences(resp.choices[0].message.content)
            result = json.loads(raw)
            tags = result["domain_tags"]
            assert isinstance(tags, list) and all(isinstance(t, str) for t in tags)
            return {"domain_tags": tags}
        except Exception as e:
            print(f"  [ERROR] {repo_key} task#{task_idx}: {e}", file=sys.stderr)
            return {"domain_tags": []}


async def _cmd_tag_async():
    """Async implementation for domain tagging."""
    data = load_benchmarks()

    total_tasks = sum(len(v["tasks"]) for v in data.values())
    print(f"Loaded {len(data)} repos, {total_tasks} tasks total.")

    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(ASYNC_MAX_CONCURRENCY)

    # Build coroutines for un-tagged tasks
    coros = []
    task_refs = []
    skipped = 0

    for repo_key, repo_data in data.items():
        repo_desc = repo_data.get("description", "")
        for idx, task_obj in enumerate(repo_data["tasks"]):
            if "domain_tags" in task_obj and task_obj["domain_tags"]:
                skipped += 1
                continue
            coros.append(
                _tag_single_task(client, semaphore, repo_key, repo_desc,
                                 task_obj["task"], idx)
            )
            task_refs.append((repo_key, idx))

    pending = len(coros)
    print(f"Already tagged: {skipped}, pending: {pending}")
    if not pending:
        print("Nothing to do.")
        return
    print(f"Tagging {pending} tasks with {MODEL} "
          f"(concurrency={ASYNC_MAX_CONCURRENCY})...")

    results = await asyncio.gather(*coros)

    # Write tags back
    success_count = 0
    error_count = 0
    all_tags = Counter()

    for (repo_key, idx), result in zip(task_refs, results):
        data[repo_key]["tasks"][idx]["domain_tags"] = result["domain_tags"]
        if result["domain_tags"]:
            success_count += 1
            for tag in result["domain_tags"]:
                all_tags[tag] += 1
        else:
            error_count += 1

    save_benchmarks(data)

    print(f"\nDone! {success_count} tagged, {error_count} errors.")
    print(f"Results written to {BENCHMARKS_FILE}")

    # Print tag distribution
    print(f"\nDomain tag distribution ({len(all_tags)} unique tags):")
    for tag, count in all_tags.most_common():
        bar = "█" * count
        print(f"  {tag:35s} {count:3d}  {bar}")


def cmd_tag():
    """Tag each benchmark task with broad domain labels."""
    asyncio.run(_cmd_tag_async())


# ═══════════════════════════════════════════════════════════════════════════════
# Step 4: Score real-world-ness
# ═══════════════════════════════════════════════════════════════════════════════

REALWORLD_SYSTEM_PROMPT = """You are an expert software engineering reviewer.
You will be given a coding task description and the repository it belongs to.

Your job is to judge how much this task feels like a **real-world software engineering problem**
that a professional developer might actually encounter on the job.

Consider these factors:
- Does the task address a practical, concrete need (not a toy/academic exercise)?
- Would a real engineering team plausibly create a ticket like this?
- Does it involve realistic constraints (APIs, compatibility, performance, security)?
- Is the scope appropriate for a real feature/bug/improvement?

Respond with ONLY valid JSON (no markdown fences):
{
  "score": <integer 1-10>,
  "reasoning": "<one or two sentences explaining your score>"
}

Scoring guide:
  1-2: Clearly artificial / toy problem with no practical use
  3-4: Somewhat contrived, unlikely to appear in a real codebase
  5-6: Plausible but generic or overly simplified
  7-8: Feels like a real engineering task with practical constraints
  9-10: Highly realistic, could be copy-pasted from an actual JIRA ticket
"""


async def _score_single_task(client, semaphore, repo_key, repo_desc, task_text, task_idx):
    """Call GPT-5.2 to score a single task. Returns parsed JSON dict."""
    async with semaphore:
        user_msg = _build_tag_user_prompt(repo_key, repo_desc, task_text)
        try:
            resp = await client.chat.completions.create(
                model=MODEL,
                temperature=0.3,
                messages=[
                    {"role": "system", "content": REALWORLD_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
            )
            raw = strip_code_fences(resp.choices[0].message.content)
            result = json.loads(raw)
            return {
                "score": int(result["score"]),
                "reasoning": result["reasoning"],
            }
        except Exception as e:
            print(f"  [ERROR] {repo_key} task#{task_idx}: {e}", file=sys.stderr)
            return {"score": -1, "reasoning": f"Error: {e}"}


async def _cmd_score_async():
    """Async implementation for real-world scoring."""
    data = load_benchmarks()

    total_tasks = sum(len(v["tasks"]) for v in data.values())
    print(f"Loaded {len(data)} repos, {total_tasks} tasks total.")

    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(ASYNC_MAX_CONCURRENCY)

    # Build coroutines for un-scored tasks
    coros = []
    task_refs = []
    skipped = 0

    for repo_key, repo_data in data.items():
        repo_desc = repo_data.get("description", "")
        for idx, task_obj in enumerate(repo_data["tasks"]):
            if "real_world_score" in task_obj and task_obj["real_world_score"] >= 0:
                skipped += 1
                continue
            coros.append(
                _score_single_task(client, semaphore, repo_key, repo_desc,
                                   task_obj["task"], idx)
            )
            task_refs.append((repo_key, idx))

    pending = len(coros)
    print(f"Already scored: {skipped}, pending: {pending}")
    if not pending:
        print("Nothing to do.")
        return
    print(f"Scoring {pending} tasks with {MODEL} "
          f"(concurrency={ASYNC_MAX_CONCURRENCY})...")

    results = await asyncio.gather(*coros)

    # Write scores back
    success_count = 0
    error_count = 0
    for (repo_key, idx), result in zip(task_refs, results):
        data[repo_key]["tasks"][idx]["real_world_score"] = result["score"]
        data[repo_key]["tasks"][idx]["real_world_reasoning"] = result["reasoning"]
        if result["score"] >= 0:
            success_count += 1
        else:
            error_count += 1

    save_benchmarks(data)

    print(f"\nDone! {success_count} scored, {error_count} errors.")
    print(f"Results written to {BENCHMARKS_FILE}")

    # Print score distribution
    scores = [r["score"] for r in results if r["score"] >= 0]
    if scores:
        avg = sum(scores) / len(scores)
        print(f"Average real-world score: {avg:.1f}")
        print("Distribution:")
        for s in range(1, 11):
            count = scores.count(s)
            bar = "█" * count
            if count:
                print(f"  {s:2d}: {bar} ({count})")


def cmd_score():
    """Score each benchmark task for real-world-ness."""
    asyncio.run(_cmd_score_async())


# ═══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════════

def print_usage():
    """Print usage help."""
    print("Usage:")
    print("  python generate_benchmarks.py generate [--batch N] [filter_key ...]")
    print("  python generate_benchmarks.py enrich [filter_key ...]")
    print("  python generate_benchmarks.py tag")
    print("  python generate_benchmarks.py score")
    print("  python generate_benchmarks.py --list-keys")


if __name__ == "__main__":
    args = sys.argv[1:]

    if not args:
        print_usage()
        sys.exit(1)

    # Utility: list all dir_name keys derived from repos.json
    if args[0] == "--list-keys":
        with open(INPUT_FILE) as f:
            urls = json.load(f)
        repos = build_repo_dict(urls)
        for key in sorted(repos.keys()):
            print(f"  {key}  ->  {repos[key]['url']}")
        sys.exit(0)

    command = args[0]
    rest = args[1:]

    if command == "generate":
        # Parse --batch N flag (default 5)
        batch_size = DEFAULT_BATCH_SIZE
        if "--batch" in rest:
            idx = rest.index("--batch")
            batch_size = int(rest[idx + 1])
            rest = rest[:idx] + rest[idx + 2:]
        filter_keys = rest if rest else None
        cmd_generate(filter_keys=filter_keys, batch_size=batch_size)

    elif command == "enrich":
        filter_keys = rest if rest else None
        cmd_enrich(filter_keys=filter_keys)

    elif command == "tag":
        cmd_tag()

    elif command == "score":
        cmd_score()

    else:
        print(f"Unknown command: {command}")
        print_usage()
        sys.exit(1)
