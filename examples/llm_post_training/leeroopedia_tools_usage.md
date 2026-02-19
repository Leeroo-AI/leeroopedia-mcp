# Leeroopedia Knowledge Base Tools

You have access to the Leeroopedia MCP tools. Use them throughout this pipeline to make informed decisions. Specifically:

- **`search_knowledge`**: Search the knowledge base before writing any code. Look up TRL SFTTrainer usage, DPO training patterns, LoRA configuration, vLLM serving config, and lm-evaluation-harness setup. Search from multiple angles in parallel.

- **`build_plan`**: Before starting each task, request a structured implementation plan from the knowledge base. Get the correct sequence of steps, key specs, and validation criteria grounded in real framework documentation.

- **`review_plan`**: After drafting your approach for each task, submit it for review against documented best practices. Catch incorrect assumptions and known pitfalls before writing code.

- **`propose_hypothesis`**: When you are unsure how to proceed or face a design decision (e.g. choosing between training strategies or serving configurations), use this tool to get ranked approaches backed by documented framework patterns.

- **`query_hyperparameter_priors`**: Query recommended values and tuning ranges for all key hyperparameters — LoRA rank/alpha, learning rates, batch sizes, DPO beta, vLLM memory utilization, tensor parallel size, and any other parameters. Get context-specific suggestions based on model size and hardware.

- **`verify_code_math`**: After writing critical code sections (training loops, serving config), verify correctness against knowledge base documentation and reference implementations.

- **`diagnose_failure`**: If any step fails or produces unexpected results, use this tool to diagnose the root cause against known failure patterns and common misconfigurations.

- **`get_page`**: When any tool response cites a `[PageID]`, retrieve the full page for deeper detail.

Use these tools proactively — do not rely solely on your training knowledge. The knowledge base contains up-to-date, verified documentation that will help you avoid common mistakes and choose optimal configurations.
