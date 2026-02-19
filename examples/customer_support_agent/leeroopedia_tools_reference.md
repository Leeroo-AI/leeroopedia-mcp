## Knowledge Base -- Leeroopedia MCP Tools

You have access to **Leeroopedia**, a curated ML/AI knowledge base, via MCP tools. These are real MCP tools registered in your environment -- call them directly like any other tool. They contain framework-specific docs, code examples, API references, and best practices.

**IMPORTANT: When you are unsure how to implement something, call these tools FIRST before guessing.** They will save you time and help you write correct code.

### Available Tools (8 total)

| Tool | What it does | Key parameters |
|------|-------------|---------------|
| `search_knowledge` | Search KB for framework docs, APIs, best practices | `query` (required), `context` (optional) |
| `build_plan` | Build a step-by-step execution plan grounded in KB | `goal` (required), `constraints` (optional) |
| `review_plan` | Review a plan against KB best practices | `proposal` (required), `goal` (required) |
| `verify_code_math` | Verify code against authoritative ML descriptions | `code_snippet` (required), `concept_name` (required) |
| `diagnose_failure` | Diagnose training/deployment failures via KB | `symptoms` (required), `logs` (required) |
| `propose_hypothesis` | Propose ranked next-step hypotheses | `current_status` (required) |
| `query_hyperparameter_priors` | Query hyperparameter values and heuristics | `query` (required) |
| `get_page` | Retrieve a specific KB page by ID | `page_id` (required) |

### When to Use These Tools

Call `search_knowledge` for questions like:
- "How do handoffs work in the OpenAI Agents SDK?"
- "How to serialize and restore agent state between requests?"
- "How to disable parallel tool calls when handoff tools are registered?"
- "How to export agent configuration to declarative JSON?"
- "How to build a router agent that classifies into 27 intent categories?"

Call `build_plan` at the start:
- "Build a multi-agent customer support triage service with 27 intent categories, handoffs, and state persistence using FastAPI"

Call `review_plan` after drafting your approach:
- Pass your implementation plan and let the KB check it against best practices

Call `diagnose_failure` when something breaks:
- Pass the error symptoms and logs to get KB-grounded debugging advice

### Example Calls

```
search_knowledge(query="OpenAI Agents SDK handoff mechanism with many specialist agents", context="building multi-agent customer support API with 27 intent categories")
search_knowledge(query="RunState serialization to_json from_json for agent state persistence")
search_knowledge(query="FastAPI SSE streaming with async generators")
build_plan(goal="multi-agent customer support triage API with 27 intent routing, handoffs, state persistence, structured output, and declarative config")
review_plan(proposal="Using OpenAI Agents SDK with handoff() for routing to grouped specialists, fine-grained intent classification in structured output", goal="production customer support triage API with 27 intents")
```
