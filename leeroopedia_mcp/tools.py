"""
MCP tool definitions for Leeroopedia agentic KG search.

Provides 8 agentic tools that run Claude Code agents to search,
read, and synthesize structured responses from the knowledge base.

The knowledge base covers 100+ ML/AI frameworks and libraries including
vLLM, SGLang, DeepSpeed, Axolotl, ROLL, MNN, ColossalAI, TRL, PEFT,
LLaMA-Factory, and many more. It contains architecture docs, API references,
config formats, best practices, and implementation patterns.
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
            "name": "search_knowledge",
            "description": """Search the knowledge base for framework documentation, API references, config formats, and best practices.

Covers a wide range of ML/AI frameworks, libraries, and tools with architecture docs,
implementation patterns, configuration references, and troubleshooting guides.

Use this tool when you need to:
- Understand how a framework, library, or API works before implementing
- Look up config formats, data structures, or expected behavior
- Learn about architecture, design patterns, or conventions of a project
- Get verified information instead of guessing about framework internals

IMPORTANT: Use this tool BEFORE you start coding whenever the task involves
a framework or library. It is much faster and more accurate than guessing.
Call this tool multiple times in parallel with different queries to search
from multiple angles at once.

Returns a synthesized answer with [PageID] citations.""",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What you want to find out (e.g., 'How does device_mapping work in ROLL?', 'What is the MNN backend selection config format?')",
                    },
                    "context": {
                        "type": "string",
                        "description": "Optional context about what you're building (e.g., 'I am implementing a CLI tool that parses YAML configs')",
                    },
                },
                "required": ["query"],
            },
        },
        {
            "name": "build_plan",
            "description": """Get a structured implementation plan based on knowledge base documentation.

Covers a wide range of ML/AI frameworks, libraries, and tools with architecture docs,
implementation patterns, configuration references, and troubleshooting guides.

Returns an actionable plan with numbered steps, specs, and validation criteria —
all based on how the framework actually works.

Use this tool when you:
- Are about to implement something and want the correct sequence of steps
- Need a plan informed by real framework documentation, not just general knowledge
- Want validation criteria to verify your implementation against

Returns: overview, key specs, numbered steps, and validation criteria.""",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "goal": {
                        "type": "string",
                        "description": "What you want to accomplish (e.g., 'Implement YAML config parser for distributed RL experiment planning')",
                    },
                    "constraints": {
                        "type": "string",
                        "description": "Optional constraints or requirements (e.g., 'Must validate GPU memory, support multi-node configs')",
                    },
                },
                "required": ["goal"],
            },
        },
        {
            "name": "review_plan",
            "description": """Review your implementation plan against knowledge base documentation before coding.

Pass your proposed approach and the KB will check it against documented best practices,
known pitfalls, and real framework behavior — catching mistakes before you write code.

Use this tool when you:
- Have a plan and want to validate it before executing
- Want to catch incorrect assumptions about how a framework works
- Need to know what pitfalls or edge cases to watch out for

Returns: approvals (what looks good), risks, and improvement suggestions.""",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "proposal": {
                        "type": "string",
                        "description": "The plan or approach to review",
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
            "description": """Verify code correctness against knowledge base documentation and reference implementations.

Checks your code against documented behavior, reference implementations,
and API contracts — catching errors before they become bugs.

Use this tool when you need:
- Verification that code correctly implements a concept, algorithm, or API contract
- Detection of logic errors, off-by-one mistakes, or wrong assumptions
- Comparison against reference implementations or documented behavior

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
                        "description": "The concept, algorithm, or API being implemented (e.g., 'GPU memory allocation for vLLM', 'YAML config validation for ROLL')",
                    },
                },
                "required": ["code_snippet", "concept_name"],
            },
        },
        {
            "name": "diagnose_failure",
            "description": """Diagnose errors, failures, or unexpected behavior using knowledge base documentation.

Checks your symptoms and logs against known failure patterns, common misconfigurations,
and documented environment issues — finding root causes faster.

Use this tool when you need:
- Root cause analysis of errors, crashes, or unexpected behavior
- Debugging configuration issues or dependency problems
- Understanding why a framework behaves differently than expected

Returns: diagnosis, fix steps, and prevention advice.""",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "symptoms": {
                        "type": "string",
                        "description": "Description of the failure symptoms or unexpected behavior",
                    },
                    "logs": {
                        "type": "string",
                        "description": "Relevant log output, error messages, or stack traces",
                    },
                },
                "required": ["symptoms", "logs"],
            },
        },
        {
            "name": "propose_hypothesis",
            "description": """Propose ranked approaches or solutions based on knowledge base documentation.

When you're unsure how to proceed, this tool suggests alternative approaches
ranked by fit — all backed by documented framework patterns and best practices.

Use this tool when you need:
- Ideas for how to implement or architect something
- Alternative approaches ranked by fit for your use case
- Suggestions backed by documented framework patterns and best practices

Returns: ranked hypotheses with rationale and suggested next steps.""",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "current_status": {
                        "type": "string",
                        "description": "Where the project stands now and what you're trying to decide",
                    },
                    "recent_experiments": {
                        "type": "string",
                        "description": "Optional description of what you've tried so far and what happened",
                    },
                },
                "required": ["current_status"],
            },
        },
        {
            "name": "query_hyperparameter_priors",
            "description": """Query documented configuration values, recommended ranges, and tuning heuristics.

Look up recommended parameter values, default settings, and tuning strategies
for frameworks and libraries — based on documented best practices.

Use this tool when you need:
- Default or recommended values for framework configuration parameters
- Recommended ranges and tuning strategies for any setting
- Context-specific suggestions (hardware, model size, task type, scale)

Returns: suggestion table with ranges and KB-grounded justification.""",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Configuration or parameter question (e.g., 'recommended GPU memory settings for vLLM serving', 'default batch size for DeepSpeed ZeRO-3')",
                    },
                },
                "required": ["query"],
            },
        },
        {
            "name": "get_page",
            "description": """Retrieve the full content of a specific knowledge base page by its exact ID.

Other tools return [PageID] citations in their responses. If you need more detail
from a cited page, call this tool with that page ID to get the full content.

Use this tool when you:
- See a [PageID] citation in a response and want the full page content
- Need deeper detail than what was included in a synthesized answer
- Want to read a specific page directly without searching

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
    "search_knowledge",
    "build_plan",
    "review_plan",
    "verify_code_math",
    "diagnose_failure",
    "propose_hypothesis",
    "query_hyperparameter_priors",
    "get_page",
}
