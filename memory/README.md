# Memory

Does the agent remember something useful after the conversation moves on?

Session history remembers the transcript. Long-term memory remembers durable
facts extracted from the transcript. This example shows the newer compact path:

```python
memory = MemoryManager(
    stores=[MemoryStoreConfig(name="main", description="learner memory", store=store)],
    namespace=("user", "alice"),
)

agent = Agent(model=model, memory_manager=memory)
```

The memory manager carries the default namespace and agent behavior, so the
agent can recall before a run and extract after a run without a separate
`MemoryConfig`.

## Examples

### [agent_memory.py](agent_memory.py)

A fully local, deterministic example. It uses:

- `MemoryManager`
- `MemoryStore`
- `InMemoryStorage`
- a tiny rule-based extractor
- a tiny model that prints whether memory reached the prompt

The first turn teaches the agent a durable preference:

```text
Remember: I prefer first-principles explanations with concrete code examples.
```

The second turn asks a new question. The example shows that the preference is
now injected into the current model input as long-term memory.

```bash
uv run memory/agent_memory.py
```

Expected shape:

```text
TURN 1 — teach a preference
  → No long-term memory was injected yet.

Stored memories
  - [preference] user.preference: I prefer first-principles explanations...

TURN 2 — ask a later question
  → I received long-term memory:
    - [preference] user.preference: I prefer first-principles explanations...
```

## Why this matters

The agent is not replaying the whole conversation to remember the preference.
It stores a compact memory entry, retrieves it before the next run, and injects
only the relevant memory context.
