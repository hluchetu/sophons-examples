# Context

What actually reaches the model.

[sessions/](../sessions/) gave the agent a memory with no ceiling: every
turn is replayed on every later call, so each request is larger than the
last until it exceeds the model's context window. A conversation manager
decides what travels.

There are three, because there are only three answers to "what do we
keep":

| Manager | Keeps | Cost |
|---|---|---|
| `NullConversationManager` | everything | nothing, until the request is rejected |
| `SlidingWindowManager` | the recent part | forgets the rest |
| `SummarizingManager` | the recent part, plus a summary | one extra model call when it fires |

Pass one to the agent and that is the whole API:

```python
agent = Agent(
    model=model,
    tools=[branch_hours],
    conversation_manager=SlidingWindowManager(max_messages=6),
)
```

Size the window in messages or in tokens — same strategy, one argument:
`SlidingWindowManager(max_tokens=400)`. A token window supplies its own
counter unless you pass a real tokenizer.

## Examples

### [context_window.py](context_window.py)

The same five-turn conversation run through each strategy, reporting the
input tokens the provider actually billed for each turn.

```bash
uv run context/context_window.py
```

Unmanaged history grows every turn and never stops. A window holds steady
by forgetting. Summarizing holds steady by compressing, at the cost of an
extra model call each time it fires.

## Two things that are not choices

**A tool call is never separated from its result.** They are one unit,
kept or dropped together. An orphaned tool result is a malformed request
that some providers reject outright, so this is an invariant rather than
a setting.

**Oversized tool results are truncated before anything is dropped.**
Shrinking a 10 KB result to its two ends loses far less than discarding
the message holding it — the agent keeps the record of what it did. Turn
it off with `truncate_tool_results=False` if you need the raw text.

## When the estimate is wrong

Sizing a window is a guess about what will fit. When the guess is wrong
the model rejects the request, and the loop asks the manager for
something smaller and retries — `reduce_context`. A sliding window
halves itself; a summarizing manager compresses below its own trigger.
Without a manager there is nothing to ask, and the run fails.
