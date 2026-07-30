# sophons-examples — Roadmap

A build order for this repo's examples, sequenced so each folder leans on
skills the previous one taught, not just topic-by-topic. Cross-checked
against what Strands, the OpenAI Agents SDK, Pydantic AI, and LangGraph
ship as their own core examples (see the comparison that produced this
list for source links).

Each phase also names the matching article in the *Architecture Patterns
Behind AI Agents* series (`~/ai-blog-next/content/articles/`), since most
patterns already have a published article but no runnable code here yet.

Voice/realtime agents are deliberately out of scope — that's its own
series (`voice-agent-architecture-patterns.mdx`, `voice-agent-prompts.mdx`).

## Phase 0 — Foundations

### 1. `tool_use/` — the bare function-calling loop
One tool, one call, no memory, no branching. Everything else in this repo
assumes you understand this loop cold before adding anything on top of it.

- Inspired by: OpenAI Agents SDK `basic/`, Strands "Weather Forecaster" / "File Operations"
- Article: `agent-patterns-react.mdx`
- Status: **done** — `basic.py` (2026-07-27). Landed as a top-level folder, not
  under `agents/`, to match the flat convention every other topic uses.

### 2. `structured_output/` — typed, validated responses
Same loop, but the agent must return a schema-shaped answer instead of
free text. Needed before anything downstream can reliably chain agents
or extract data from a response.

- Inspired by: Strands "Structured Output", Pydantic AI (`pydantic_model.py`, `sql_gen.py`, `flight_booking.py`) — this is basically Pydantic AI's whole premise
- Status: **done** — `basic.py`, `with_tools.py` (2026-07-27). Needed an SDK
  change first: `Agent(output_type=...)` in sophons, implemented as an
  `OutputTool` whose argument schema is the model's JSON schema, intercepted
  by the loop rather than executed. Tool calling was forced rather than
  chosen — DeepSeek rejects `response_format={"type": "json_schema"}`.
  Known gap: `result.output` is typed `Any`, so editors cannot infer the
  concrete type; generics on `Agent` would fix it.

## Phase 1 — State & Knowledge

### 3. `sessions/` — does the agent remember this conversation?
Conversation history that survives across turns, and across restarts.
The session id is what enables persistence; the session manager only
decides where the history lives.

- Builds on: `tool_use/` (same loop, now stateful)
- Inspired by: OpenAI Agents SDK `memory/` (SQLite/Redis/SQLAlchemy/file-backed), Strands "Session Management"
- Article: `agent-memory-patterns.mdx`
- Status: **done** — `session.py` (2026-07-27)

### 3b. `context/` — what actually reaches the model?
Session memory has no ceiling: every turn is replayed on every later call
until the request exceeds the context window. A conversation manager
decides what travels — everything, the recent part, or a summary of the
rest.

- Builds on: `sessions/` (the history this trims)
- Inspired by: Strands "Conversation Management"
- Status: **withdrawn, needs a redesign.** The SDK side is done and good —
  three managers instead of six, `SummarizingManager` fixed (it had never
  worked; it called a method no model implements), tool pairs never split,
  `reduce_context` for overflow. See
  `sophons/docs/context_management_architecture.md`.

  The *example* was removed on 2026-07-29 because it did not demonstrate
  its own claim. It reported `metrics.input_tokens` per turn, which is the
  **sum of every model call in that turn** — so a turn that used a tool
  counted two calls and a turn that did not counted one. That variation
  swamped the effect of trimming, and since the agent's tool decisions are
  non-deterministic the output changed shape run to run. Observed:

      no manager        693  833  945  487   573
      sliding window    693  833  938  479  1011

  Unmanaged did not grow, and the managed run finished *larger*.

  The fix identified: **drop the tool from the example.** With no tools each
  turn is exactly one model call, so `input_tokens` becomes the context size
  measured by the provider — exact, repeatable, no hook required. Put the
  facts in the system prompt and ask follow-ups about them. Tools belong to
  `tool_use/`; here they are noise.

### 3c. `memory/` — does it remember you across conversations?
The first example to use `sophons.memory` — extraction, namespaces,
retrieval, reflection. Change the session id and a session-only agent
forgets you completely; this is what survives that boundary.

- Builds on: `sessions/` (the boundary it crosses)
- Inspired by: Mem0's add/search convention, which `MemoryManager` follows; LangGraph's store-vs-checkpointer split
- Article: `what-moves-where-agent-memory.mdx`
- Status: **done** — `customer_support.py` (2026-07-29). Needed an SDK pass
  first, and a larger one than expected: `Agent(memory_manager=...)` makes
  memory first-class, `MemoryManager(namespace=..., inject_limit=...)` binds
  the namespace at construction, and retrieve-inject-extract now happens
  inside the loop. Also added `sophons.tools.memory` so memory is reachable
  as a tool the agent calls, and a SQLite storage backend alongside
  `InMemoryStorage`. The example is one line of wiring as a result:
  `Agent(model=model, memory_manager=memory)`.

### 4. `rag/` — retrieval as grounding
Naive → hybrid (dense + BM25 + RRF) → reranking → query rewriting →
corrective retry. Retrieval is memory's external counterpart: instead of
remembering, you look things up.

- Builds on: `memory/` (assembling context), `agents/structured_output/` (grounded, citation-formatted answers)
- Inspired by: every framework's `rag` example, plus your own `RAG_SERIES_PLAN.md`
- Status: **mostly done** — `naive.py`, `hybrid.py`, `reranker.py`, `query_rewriting.py`, `corrective.py` all exist; audited 2026-07-26, fix queue in the Review Log below

## Phase 2 — Decision-Making

### 5. `planning/` — decompose before you act
Plan-and-execute: break a task into ordered steps up front, then run the
plan, instead of deciding tool-by-tool.

- Builds on: `agents/tool_use/` + `memory/` (a plan is state you track across steps)
- Inspired by: LangGraph `plan-and-execute`, `rewoo`, `lats`
- Article: `agent-patterns-planning.mdx`
- Status: **not started**

### 6. `routing/` — one decision, then dispatch
Classify the incoming request and send it to the right tool, prompt, or
sub-flow. The simplest possible "decide, then act" pattern — a
degenerate one-step plan.

- Builds on: `agents/structured_output/` (the router's decision needs a typed shape), `planning/`
- Inspired by: OpenAI Agents SDK `agent_patterns` (routing example)
- Article: `agent-patterns-routing.mdx`
- Status: **not started**

## Phase 3 — Multiple Agents

### 7. `multi_agent/` — orchestrator + specialists, and handoffs
Generalize routing from "pick a tool" to "pick another agent, and hand it
the conversation." Covers both agents-as-tools and full context handoff.

- Builds on: `routing/` (routing generalized past single-hop)
- Inspired by: OpenAI Agents SDK `handoffs`, Pydantic AI `medical_agent_delegation.py`, LangGraph `multi_agent`, Strands "Multi-Agent Example"
- Article: `agent-patterns-multi-agent.mdx`
- Status: **not started** (`agents/agentic_RAG/` is an early taste of this, worth revisiting once this folder exists)

### 8. `reflection/` — a second pass reviews the first
A critic (a second agent, or a second pass) checks and revises the
original output before it's returned. The natural next thing to build
once you have two agents talking to each other.

- Builds on: `multi_agent/` (critic-as-second-agent)
- Inspired by: LangGraph `reflection`/`reflexion`, OpenAI Agents SDK "LLM as a judge" pattern
- Article: `agent-patterns-reflection.mdx`
- Status: **not started** (shares machinery with `evaluation/`'s LLM-as-judge)

## Phase 4 — Guarding, Watching, Measuring

These wrap around whatever you've built above, so they land after the
agent patterns exist, not before.

### 9. `guardrails/` — block unsafe actions before they happen
Tool permission policies, PII redaction, human approval gates.

- Article: `agent-patterns-guardrails.mdx`
- Status: **done** — `approval_chat.py`, `guarded_agent.py`, `human_approval.py`; audited 2026-07-26, no issues found

### 10. `observability/` — trace the whole run
Once there are multiple moving parts (tools, retrieval, sub-agents,
critics), you need to see what actually happened, end to end.

- Article: `agent-patterns-observability.mdx`
- Status: **done** — `traced_agent.py`; extend span coverage once `planning/`, `routing/`, `multi_agent/` exist; audited 2026-07-26, small polish queue in the Review Log below

### 11. `evaluation/` — judge the output, not just watch it run
LLM-as-judge, faithfulness (decompose-then-verify), pass@k — currently
scoped to RAG, worth extending to routing/multi-agent/reflection outputs
once those exist.

- Article: `agent-patterns-evaluation.mdx`
- Status: **partially done** — `faithfulness.py`, `pass_k.py` clean (audited 2026-07-26); `rag.py` shares the `rag/` fix queue below

## Phase 5 — Ship It

### 12. `capstone/` — one real vertical demo
Wire tool_use + structured_output + memory + rag + guardrails +
multi_agent together into one demo people would actually recognize as
"an agent" — e.g. a support-ticket triage agent, reusing the existing
bank-support docs corpus already in `rag/docs/`.

- Builds on: everything above
- Inspired by: Pydantic AI `bank_support.py` / `flight_booking.py`, LangGraph `customer-support`, OpenAI Agents SDK `customer_service`
- Status: **not started**

### 13. `deployment/` — run it as a real service
Containerize the capstone, then ship it to one real target — start with
a single managed target (Fargate or Lambda) rather than covering every
option. The part every framework treats as separate from "agent
patterns": env config, secrets, logging, cold starts.

- Builds on: `capstone/`
- Inspired by: Strands' full deployment tier (Docker → Lambda/Fargate/App Runner → EC2/EKS/Kubernetes)
- Status: **not started** — explicitly in scope, don't skip this phase

## Deliberately deferred / not a top-level folder

- **Extraction** — folded into `agents/structured_output/` as a variant, not its own folder (it's the same schema-constrained-output skill applied to a document instead of a chat turn).
- **Voice/realtime** — already its own series (see above), not duplicated here.
- **Web navigation / computer use / code assistant** — specialized tool integrations on top of `agents/tool_use/`; worth a follow-up folder once the fundamentals above are solid, not before.

## Review Log

### 2026-07-26 — line-by-line audit of Phase 4 (`rag/`, `evaluation/`, `guardrails/`, `observability/`)

All 12 existing example files read line by line before starting new Phase 0
work, per the "review what's already done first" call. Findings:

- **Dead `asyncio.iscoroutine(response)` check, copy-pasted into 4 files**
  (`rag/naive.py`, `rag/corrective.py`, `rag/chat.py`, `evaluation/rag.py`):
  `DeepSeekModel.invoke()` (`sophons/integrations/models/deepseek.py`) is
  always synchronous, so the branch never fires. Misleads a line-by-line
  reader into thinking `invoke()` might be async. Fix queued: remove the
  check everywhere; `naive.py`'s `ask()` needs no `await` at all once it's
  gone, and since that's the only await in the file, `asyncio` can drop out
  of `naive.py` entirely.
- **`rag/naive.py`'s footer only cites `chunks[0]`**, but the README's own
  sample output shows multiple sources (`reversals.md#chunk_0,
  branches.md#chunk_1, ...`). `corrective.py` already joins all retrieved
  sources — match that.
- **`observability/traced_agent.py`** is missing the module docstring every
  other example has (what it demonstrates + a `Run:` block), and its
  `deepseek_api_key` field has a `= ""` default where every other file
  makes it required — masks a missing key until it fails deep inside the
  OpenAI client instead of failing fast at startup via pydantic validation.
- `guardrails/` (all 3 files) and `evaluation/faithfulness.py` /
  `evaluation/pass_k.py` — clean, no issues found.

Fixes identified but **not yet applied** — queued for a separate pass, not
bundled into this roadmap update.
