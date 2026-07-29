# Memory

Long-term memory is what lets an agent adapt after the current conversation
moves on. Customer support is the natural example: a useful support agent should
remember the customer's plan, preferences, and recent support history without
the developer manually rebuilding that context on every request.

Session history answers:

```text
What did we just say in this chat?
```

Long-term memory answers:

```text
What durable fact, preference, decision, or study event should the agent
remember later?
```

Sophons keeps the setup small:

```python
store = MemoryStore(
    storage=InMemoryStorage(),
    retrievers=[LexicalRetriever()],
)

memory = MemoryManager(
    stores=[MemoryStoreConfig(name="main", description="customer memory", store=store)],
    extractor=LLMMemoryExtractor(model=model),
    namespace=("customer", "C-1042"),
)

agent = Agent(model=model, memory_manager=memory)
```

The agent now uses memory automatically:

```text
before the model call -> retrieve relevant memories
during the model call -> inject them into context
after the model call  -> extract new memories when an extractor is configured
```

## Examples

### [customer_support.py](customer_support.py)

A model-backed customer support example that uses automatic extraction.

The first turn gives the agent durable customer context:

```text
Please remember this for customer C-1042: I am on the Luche Business plan,
I prefer short answers with clear next steps, and my supplier payment on card
ending 4481 failed yesterday.
```

Sophons extracts useful memories after that run. Then a later turn asks:

```text
The supplier payment failed again. Can you check the card ending 4481 and tell
me what to do next?
```

The developer does not manually create memory entries or paste the customer
context into the prompt. Sophons extracts the memory, stores it, retrieves it
later, and injects it into the agent run. The agent can still use tools for
live facts, like checking card status.

```bash
uv run memory/customer_support.py
```

## What Sophons Is Hiding For You

Without the SDK, every app has to reinvent the same glue:

```text
choose a memory namespace
search the memory store
format retrieved memories
insert them into the prompt
keep storage and retrieval separate
avoid mixing users' memories
wire memory into the agent loop
```

With Sophons, that becomes:

```python
agent = Agent(model=model, memory_manager=memory)
```
