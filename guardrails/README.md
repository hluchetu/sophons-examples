# Guardrails

Examples of `sophons.guardrails` — checks that stand in front of the agent
and say no *before* the action happens. Evaluation judges runs after the
fact; guardrails are for the actions you cannot take back.

## Examples

### [guarded_agent.py](guarded_agent.py)

The same refund agent, with and without guardrails, facing the same two
hostile requests: an over-limit refund and a message containing a card
number. The final tally is the argument:

```
money refunded unguarded: 280.00
money refunded guarded:   30.00
```

The guarded agent's `ToolPermissionGuardrail` blocks the over-limit
`refund_order` call before it executes — the model receives the reason and
explains the limit to the user instead. The `PatternGuardrail` redacts the
card number from the input before the model ever sees it.

```bash
uv run guardrails/guarded_agent.py
```

### [human_approval.py](human_approval.py)

Human in the loop at the terminal: refunds up to 100.00 run unattended;
anything larger pauses the run and asks the person at the keyboard —
the Claude Code permission-prompt pattern.

```
[approval needed] refund of 250.0 exceeds the auto-approve limit of 100.0
  -> refund_order({'order_id': 'ord_42', 'amount': 250.0})
Approve? [y/N]
```

Deny it and the model is told who said no — it explains and offers
alternatives instead of crashing.

```bash
uv run guardrails/human_approval.py
```

### [approval_chat.py](approval_chat.py)

The interactive version: a rich chat CLI (panels, history, metrics) where
approval requests render as panels inside the conversation and you decide
live. Built on the `CallbackApprover` seam — the approver is just an async
function that draws a panel and reads your answer.

```bash
uv run guardrails/approval_chat.py
```
