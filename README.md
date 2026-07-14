# sophons-examples

Runnable examples for [Sophons](https://github.com/hluchetu/sophons) — an agent
and RAG framework built from first principles.

Each top-level folder is one topic. Every example is a small, self-contained
script with its own README explaining what it demonstrates and how to run it.

| Folder | What it shows |
|---|---|
| [observability/](observability/) | Agents traced end-to-end with OpenTelemetry — console spans, OTLP export, the full span tree of a run |
| [evaluation/](evaluation/) | Judging agent answers with `sophons.evals` — LLM-as-judge verdicts, faithfulness via decompose-then-verify |
| [guardrails/](guardrails/) | Blocking unsafe actions before they happen — tool permission policies, PII redaction, guarded vs unguarded |
| [rag/](rag/) | Retrieval-augmented generation stage by stage — local embeddings, vector search, grounded answers |

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
