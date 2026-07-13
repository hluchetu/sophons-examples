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
