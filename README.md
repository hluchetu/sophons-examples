# sophons-examples

Runnable examples for [Sophons](https://github.com/hluchetu/sophons) — an agent
and RAG framework built from first principles.

Each top-level folder is one topic. Every example is a small, self-contained
script with its own README explaining what it demonstrates and how to run it.

The table reads in build order — each folder leans on the one above it.

| Folder | What it shows |
|---|---|
| [tool_use/](tool_use/) | The function-calling loop on its own — one tool, one call, the model's choice and arguments made visible |
| [structured_output/](structured_output/) | Typed, validated answers via `output_type` — a Pydantic model instead of prose, carried over the tool-calling channel |
| [sessions/](sessions/) | Whether the agent remembers this conversation — the session id is the switch, the session manager only decides where history lives |
| [memory/](memory/) | Long-term memory — remember customer context and inject it into later support runs |
| [memory/](memory/) | What the agent remembers after the conversation ends — durable facts extracted, stored under a namespace, and recalled in a later one |
| [rag/](rag/) | Retrieval-augmented generation stage by stage — local embeddings, vector search, grounded answers |
| [guardrails/](guardrails/) | Blocking unsafe actions before they happen — tool permission policies, PII redaction, guarded vs unguarded |
| [observability/](observability/) | Agents traced end-to-end with OpenTelemetry — console spans, OTLP export, the full span tree of a run |
| [evaluation/](evaluation/) | Judging agent answers with `sophons.evals` — LLM-as-judge verdicts, faithfulness via decompose-then-verify |

## Setup

Uses [uv](https://docs.astral.sh/uv/). Sophons is installed as an editable
dependency from the sibling checkout (it is not on PyPI yet) — see
`[tool.uv.sources]` in `pyproject.toml`.

```bash
uv sync
```

Model-backed examples need an API key — put it in a `.env` file at the repo
root (untracked):

```
DEEPSEEK_API_KEY=sk-...
```

Then run any example with uv:

```bash
uv run observability/traced_agent.py
```
