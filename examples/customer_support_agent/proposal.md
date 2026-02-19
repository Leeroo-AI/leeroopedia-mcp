# Demo Proposal: Multi-Agent Customer Support Triage Service

## Overview

Build a production-style customer-support triage API. The service runs a multi-agent team where specialist agents handle different ticket categories, delegate work to each other via explicit handoffs, and pause for human approval before taking irreversible actions like issuing refunds.

The service must expose an HTTP API, persist conversation state between requests, and return structured JSON suitable for writing into a ticketing system.

Use a Python multi-agent framework -- do not build the orchestration from scratch using raw LLM API calls.

---

## Environment

You are running inside the `task1` conda environment. You are free to install any Python packages you need using `pip install` or `conda install`. Install whatever is required to complete the pipeline.

---

## Task -- Triage API with Handoffs, Approval Gates, and State Persistence

Implement a FastAPI service that orchestrates a multi-agent support team. The service must classify each incoming ticket into one of **27 fine-grained intent categories** and route it to the appropriate specialist agent for resolution.

### Agent Architecture

You must build at minimum:

1. **Router Agent** -- Receives the customer message, classifies it into one of the 27 intent categories below, and hands off to the correct specialist. Does not attempt to resolve issues itself.
2. **Specialist Agents** -- You may organize specialists however you want (e.g. one agent per intent, or grouped by topic area). Each specialist handles its assigned intents and produces a structured resolution.

A reasonable grouping might be:
- **Refunds Specialist** -- get_refund, check_refund_policy, track_refund
- **Orders Specialist** -- cancel_order, place_order, change_order
- **Payments Specialist** -- payment_issue, check_invoice, get_invoice, check_payment_methods, check_cancellation_fee
- **Accounts Specialist** -- create_account, delete_account, edit_account, switch_account, recover_password, registration_problems
- **Support Specialist** -- contact_customer_service, contact_human_agent, complaint, review
- **Shipping Specialist** -- track_order, set_up_shipping_address, change_shipping_address
- **Delivery Specialist** -- delivery_options, delivery_period
- **Subscriptions Specialist** -- newsletter_subscription

But the critical requirement is: **the router must output the exact intent category, not a broad group.** The structured output's `category` field must be one of the 27 intent values listed below.

### The 27 intent categories

The router must classify each ticket into exactly one of these intents:

| Group | Intent categories |
|---|---|
| Refund-related | `get_refund`, `check_refund_policy`, `track_refund` |
| Order-related | `cancel_order`, `place_order`, `change_order` |
| Payment-related | `payment_issue`, `check_invoice`, `get_invoice`, `check_payment_methods`, `check_cancellation_fee` |
| Account-related | `create_account`, `delete_account`, `edit_account`, `switch_account`, `recover_password`, `registration_problems` |
| Support-related | `contact_customer_service`, `contact_human_agent`, `complaint`, `review` |
| Shipping-related | `track_order`, `set_up_shipping_address`, `change_shipping_address` |
| Delivery-related | `delivery_options`, `delivery_period` |
| Subscription-related | `newsletter_subscription` |

**Important:** Many intents are semantically close (e.g. `check_invoice` vs `get_invoice`, `change_order` vs `cancel_order`, `contact_customer_service` vs `contact_human_agent`). The router must make fine-grained distinctions based on the message text alone. Getting the broad group right is not enough -- the exact intent must be correct.

### Core Requirements

**Handoff-based routing:**
Agent routing must use the framework's native handoff mechanism so the next speaker is selected from explicit handoff events. Do NOT implement routing by parsing free-form agent text output or using a simple if/else classifier.

**Human approval gate:**
When any agent needs to perform an irreversible action (refund, account deletion, escalation to engineering), the multi-agent run must pause cleanly. The API returns a response indicating approval is needed. On the next request, the client submits approval/rejection, and the team resumes with the same conversation state. Do NOT block inside a single continuous run with `input()` or `time.sleep()` to wait for human input.

**State persistence:**
After each exchange, serialize the full team state (agents, conversation history, pending actions) to disk using the framework's built-in state serialization. On the next request for the same ticket, restore from the saved state rather than replaying the full transcript into a fresh team.

**Streaming events:**
The API streams intermediate agent events (which agent is speaking, tool calls, handoff events) to the client via Server-Sent Events (SSE), so the frontend can show real-time progress.

**Structured output:**
The final resolution must be a validated structured object (not plain text) containing: `ticket_id`, `category`, `resolution_summary`, `actions_taken`, `requires_followup`. The `category` field must be one of the 27 intent categories listed above. Invalid model outputs must be caught and handled gracefully before returning.

**Critical -- intent passthrough:**
If you group specialists (e.g. one "Payments Specialist" handling 5 payment intents), the router's classified intent must survive the handoff and appear in the final structured output's `category` field. This requires a **programmatic mechanism** -- for example, storing the intent in a shared context object during an `on_handoff` callback, passing it as an argument to the handoff tool, or having the specialist output it in a structured JSON block that the service code parses. Do NOT rely on the intent being mentioned as free-form text in the conversation history and attempting to extract it later -- this is fragile and commonly fails.

Three specific anti-patterns to avoid:
1. Do NOT infer the fine-grained intent from the specialist agent's name -- that only gives you the broad group (e.g. "payments"), not the exact intent (e.g. "check_invoice" vs "get_invoice" vs "payment_issue").
2. Do NOT assume the intent will be available in the shared context unless your code explicitly writes it there -- agent instructions saying "store the intent" do not cause a programmatic variable to be set. Verify that `ctx.classified_intent` (or equivalent) is actually populated after a handoff by testing it.
3. Do NOT use a keyword-based fallback classifier that defaults to a single catch-all category (e.g. `contact_customer_service`) when the primary classification is unavailable. If a fallback is needed, it must cover all 27 intents with reasonable accuracy and must not systematically bias toward any one category.

**Verify the full pipeline before testing:** Send 3-5 sample tickets from different intent groups through the running service and confirm the `category` field in each response matches the expected intent. A common failure mode is that the router correctly routes to the right specialist, but the service code cannot recover the fine-grained intent from the agent output -- this results in correct routing but wrong `category` in the API response.

**Declarative configuration:**
The full agents/team/tools setup must be exportable to a single JSON configuration file and reloadable from that JSON -- so the same team can be recreated without running setup code. Include a comment or warning that declarative configs should only be loaded from trusted sources.

**Parallel tool calls disabled:**
The model client must explicitly disable parallel tool/function calls when agent-to-agent handoff tools are registered, to prevent the model from simultaneously handing off and calling other tools.

---

## Test Data

A pre-built test corpus file `test_corpus.json` is provided in your working directory. **Use it directly -- do NOT rebuild or modify it.**

The corpus contains 200 support tickets (~7-8 per intent across 27 intent categories) derived from the Bitext Customer Support Intent Dataset. Each ticket has:

```json
{
  "ticket_id": "T-001",
  "message": "<customer message>",
  "expected_category": "get_refund"
}
```

The 27 possible values for `expected_category` are: `get_refund`, `check_refund_policy`, `track_refund`, `cancel_order`, `place_order`, `change_order`, `payment_issue`, `check_invoice`, `get_invoice`, `check_payment_methods`, `check_cancellation_fee`, `create_account`, `delete_account`, `edit_account`, `switch_account`, `recover_password`, `registration_problems`, `contact_customer_service`, `contact_human_agent`, `complaint`, `review`, `track_order`, `set_up_shipping_address`, `change_shipping_address`, `delivery_options`, `delivery_period`, `newsletter_subscription`.

Note: there is NO separate `intent` field. The `expected_category` IS the intent. Your service must determine the correct intent purely from the message text.

### Rules

- Build the software first, then send all 200 test tickets (from `test_corpus.json`) through the running service.
- Do NOT use the test set to tune, debug, or optimize the software. The test set is for reporting only.
- If there are technical issues that prevent producing results on some tickets, that is acceptable -- report what you can.

---

## Completion Requirement

Do NOT stop until the full pipeline has finished and you have reported the results of running the test corpus through the service. The pipeline must run end-to-end without interruption: build the service, start it, send all 200 test tickets (from `test_corpus.json`) through it, and report the results. Only consider the task done when you have printed:

1. How many of the 200 tickets were successfully processed.
2. The intent classification breakdown -- how many tickets were classified into each of the 27 intent categories.
3. A sample resolution object from one completed ticket.
4. Whether the service started, accepted requests, and returned structured responses without crashing.
