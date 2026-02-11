"""
MCP tool definitions for Leeroopedia agentic KG search.

Provides 7 agentic tools that run Claude Code agents to search,
read, and synthesize structured responses from the knowledge base.
"""

from typing import Any, Dict, List


def get_tool_definitions() -> List[Dict[str, Any]]:
    """
    Get MCP tool definitions for agentic KG search.

    Returns:
        List of tool definition dictionaries.
    """
    return [
        {
            "name": "consult_literature",
            "description": """Search the curated ML/AI knowledge base like a research librarian.

An AI agent searches from multiple angles, reads relevant pages, and synthesizes
a consensus answer grounded in knowledge base evidence.

Use this tool when you need:
- Foundational understanding of an ML/AI concept
- Synthesis across multiple related topics
- Verified, cited answers from a curated knowledge base

Returns a synthesized summary with [PageID] citations, distinguishing
between established consensus and emerging ideas.""",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Research question or topic to investigate",
                    },
                    "context": {
                        "type": "string",
                        "description": "Optional additional context to guide the search",
                    },
                },
                "required": ["query"],
            },
        },
        {
            "name": "build_plan",
            "description": """Build a step-by-step ML execution plan grounded in knowledge base evidence.

An AI agent searches Workflows, Principles, Implementations, and Heuristics
to construct a detailed plan for achieving the specified goal.

Use this tool when you need:
- A structured plan for an ML task (fine-tuning, training, deployment, etc.)
- Step-by-step instructions grounded in verified practices
- Specs, requirements, and validation criteria

Returns: overview, key specs, numbered steps, and tests/validation criteria.""",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "goal": {
                        "type": "string",
                        "description": "What you want to accomplish",
                    },
                    "constraints": {
                        "type": "string",
                        "description": "Optional constraints or requirements (e.g., hardware limits, time budget)",
                    },
                },
                "required": ["goal"],
            },
        },
        {
            "name": "review_plan",
            "description": """Review a proposed ML plan against knowledge base best practices.

An AI agent searches for best practices, known pitfalls, and relevant heuristics
to evaluate the proposed plan.

Use this tool when you need:
- Validation of a plan before execution
- Identification of risks and pitfalls
- Improvement suggestions backed by KB evidence

Returns: approvals (what looks good), risks, and suggestions.""",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "proposal": {
                        "type": "string",
                        "description": "The plan or proposal to review",
                    },
                    "goal": {
                        "type": "string",
                        "description": "The intended goal of the plan",
                    },
                },
                "required": ["proposal", "goal"],
            },
        },
        {
            "name": "verify_code_math",
            "description": """Verify code correctness against authoritative ML/math concept descriptions.

An AI agent searches for authoritative concept descriptions and reference
implementations, then checks the provided code for mathematical correctness.

Use this tool when you need:
- Verification that code correctly implements a mathematical concept
- Detection of numerical or algorithmic errors
- Comparison against reference implementations

Returns: verdict (Pass/Fail), analysis of discrepancies.""",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "code_snippet": {
                        "type": "string",
                        "description": "The code to verify",
                    },
                    "concept_name": {
                        "type": "string",
                        "description": "The mathematical/ML concept being implemented",
                    },
                },
                "required": ["code_snippet", "concept_name"],
            },
        },
        {
            "name": "diagnose_failure",
            "description": """Diagnose ML training or deployment failures using knowledge base evidence.

An AI agent searches Heuristics for known failure patterns and Environment
pages for dependency or configuration issues.

Use this tool when you need:
- Root cause analysis of training failures, NaN losses, OOM errors, etc.
- Environment or dependency troubleshooting
- Prevention advice based on documented patterns

Returns: diagnosis, fix steps, and prevention advice.""",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "symptoms": {
                        "type": "string",
                        "description": "Description of the failure symptoms",
                    },
                    "logs": {
                        "type": "string",
                        "description": "Relevant log output or error messages",
                    },
                },
                "required": ["symptoms", "logs"],
            },
        },
        {
            "name": "propose_hypothesis",
            "description": """Propose ranked research hypotheses grounded in knowledge base evidence.

An AI agent searches for alternative approaches, strategies, and relevant
principles to suggest next steps for an ML project.

Use this tool when you need:
- Ideas for what to try next in an ML project
- Alternative approaches ranked by promise
- Experiment suggestions backed by KB evidence

Returns: ranked hypotheses with rationale and suggested experiments.""",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "current_status": {
                        "type": "string",
                        "description": "Where the project stands now",
                    },
                    "recent_experiments": {
                        "type": "string",
                        "description": "Optional description of recent experiments and their outcomes",
                    },
                },
                "required": ["current_status"],
            },
        },
        {
            "name": "query_hyperparameter_priors",
            "description": """Query documented hyperparameter values, ranges, and tuning heuristics.

An AI agent searches the knowledge base for documented hyperparameter values,
recommended ranges, and tuning strategies.

Use this tool when you need:
- Starting values for hyperparameters (learning rate, batch size, etc.)
- Recommended ranges and tuning strategies
- Context-specific suggestions (model size, task type, hardware)

Returns: suggestion table with ranges and KB-grounded justification.""",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Hyperparameter question or topic (e.g., 'learning rate for LoRA fine-tuning Llama-3 8B')",
                    },
                },
                "required": ["query"],
            },
        },
        {
            "name": "get_page",
            "description": """Retrieve the full content of a specific knowledge base page by its exact ID.

A direct lookup — no AI agent needed. Returns the complete page content
including type, overview, full content, domains, sources, and related pages.

Use this tool when you:
- Already know the exact page ID (e.g., from a previous search result citation)
- Want to read a specific page without searching
- Need the full content of a page referenced by another tool

Returns the page formatted as markdown, or an error if the page ID is not found.""",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "page_id": {
                        "type": "string",
                        "description": "Exact page ID to retrieve (e.g., 'Workflow/QLoRA_Finetuning', 'Principle/LoRA_Rank_Selection')",
                    },
                },
                "required": ["page_id"],
            },
        },
    ]


TOOL_NAMES = {
    "consult_literature",
    "build_plan",
    "review_plan",
    "verify_code_math",
    "diagnose_failure",
    "propose_hypothesis",
    "query_hyperparameter_priors",
    "get_page",
}
