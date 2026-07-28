# Sessions

Does the agent remember *this* conversation?

A session is the transcript: the messages of one continuing exchange,
reloaded before each run and saved after. It is not long-term memory —
change the session id and the agent has never met you. That boundary is
what [memory/](../memory/) is for.

## Examples

### [session.py](session.py)

The same two turns, three times. The second turn is the test:

> Is it still open on Saturdays?

"It" has no referent. Either the previous turn is in context or the
question is unanswerable.

```
FORGETFUL — no session id
  → "Could you please tell me which branch you're interested in?"

IN PROCESS — session id, default manager
  → "Yes, the Westlands branch is open on Saturdays — until 12:00 (noon)."

ON DISK — session id + FileSessionManager
  → same answer, and written to .sessions/branch-chat.json
```

**The session id is the switch, not the session manager.** `Agent`
substitutes an `InMemorySessionManager` when none is given, so an agent
built with no manager at all still remembers — as long as it is passed a
`session_id`. The middle run proves it: a fresh `Agent`, no manager,
correct answer.

What the manager decides is *where* the history lives:

| session_manager | session_id | behaviour |
|---|---|---|
| omitted | `None` | forgets every turn |
| omitted | `"branch-chat"` | remembers, until the process exits |
| `FileSessionManager` | `"branch-chat"` | remembers across restarts |

```bash
uv run sessions/session.py
```

## What gets saved

Open `.sessions/branch-chat.json` after a run and it is exactly what it
looks like — a JSON array of messages, reloaded and prepended next time.
No embeddings, no state machine.

Two details worth noticing. Only `user` and `assistant` messages are
stored, so a tool call is replayed as its *conclusion* rather than as a
call. And every message carries a stable id, which is what lets a stored
turn be joined to a trace span or addressed for deletion.

The file grows on every run, without limit. That is the problem
[context/](../context/) solves.
