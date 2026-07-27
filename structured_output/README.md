# Structured Output

Same loop as [tool_use/](../tool_use/), but the agent must return a shape
instead of a sentence. Pass `output_type=SomeModel` and the run ends with a
validated instance on `result.output`.

The shape travels over the tool-calling channel: Sophons declares a tool
named `structured_response` whose *argument schema* is your model's JSON
schema, and intercepts the call rather than executing it. Nothing about the
Python class reaches the model — only JSON Schema, in the same `tools`
array a real tool would use.

## Examples

### [basic.py](basic.py)

Two support messages triaged into `Ticket(category, urgency, summary)`:

```
You    The ATM at Westlands swallowed my card and I fly out tomorrow!
Tool   Ticket(category='cards', urgency=5)
Agent  Customer's card was swallowed by the Westlands ATM and they have a
       flight tomorrow, requiring urgent assistance.
       cards · urgency 5/5 · steps=1

You    hi, just wondering what the fee code FEE-WDR-021 is about
Tool   Ticket(category='other', urgency=1)
Agent  Customer is asking about the meaning of fee code FEE-WDR-021.
       other · urgency 1/5 · steps=1
```

`result.output` is a real `Ticket`, not a string that looks like one —
`ticket.urgency` is an `int` you can compare, sort, and route on.

Note what the system prompt does *not* say. It never lists the categories
or explains the urgency scale; it is one line, "You triage incoming Luche
Bank support messages." The category list and the 1–5 range live in
`Field(description=...)`, which Pydantic writes into the JSON schema. **The
schema is the prompt.**

```bash
uv run structured_output/basic.py
```

### [with_tools.py](with_tools.py)

Structured output composed with a real tool. Triage now needs a fact the
model cannot know — whether the account is premium — so the run has two
beats:

```
You    Account 0114455 here — the ATM at Westlands swallowed my card
       and I fly out tomorrow!

Tool   account_tier({'account_number': '0114455'})
Tool   structured_response({'account_number': '0114455', 'tier': 'premium',
       'category': 'cards', 'urgency': 5, 'summary': "Customer's card was
       swallowed by the ATM at Westlands branch..."})

Agent  Customer's card was swallowed by the ATM at Westlands branch and
       they have a flight tomorrow, requiring urgent resolution.
       0114455 · premium · cards · urgency 5/5 · steps=2
```

Both calls print from the same `result.tool_uses` list, because to the
model they *are* the same kind of thing. The difference is entirely on the
Sophons side: `account_tier` gets executed and its result fed back, while
`structured_response` is intercepted, validated, and ends the run.

`steps=2` is the tell — one round trip to look up the tier, a second to
fill in the shape once the answer is known.

```bash
uv run structured_output/with_tools.py
```

## Validation and retries

If the model returns arguments that fail validation, the error is handed
back to it as a failed tool result so it can correct itself:

```
Your arguments did not match the required schema. Fix these problems and
call the tool again:
- category: Field required
- urgency: Input should be less than or equal to 5
```

Those retries are bounded by the run's `max_steps`, not a separate budget.

Worth knowing: the constraints in your schema (`ge=1, le=5` → `minimum` /
`maximum`) are sent to the model, but DeepSeek does not do constrained
decoding on tool arguments — nothing at the model level *forbids*
`urgency: 99`. Validation is what makes the constraint real. Providers with
strict schema enforcement close that gap; tool calling alone does not.
