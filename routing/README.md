# Routing

One decision, then dispatch.

In [tool_use/](../tool_use/) the model decides and acts in the same breath —
it picks a tool and the loop runs it. Routing pulls those apart. A classifier
answers one question, *who should handle this*, and returns it as a value.
Something else acts on that value.

The point is that the decision becomes inspectable. You can log it, count it,
alert on it, and test the classifier without running any of the specialists.
"How many fraud reports came in today" stops being something you infer from
answers and becomes a field you already have.

## The pattern

Routing needs nothing from Sophons beyond `output_type`. A decision is a
shape, so it is the same mechanism as
[structured_output/](../structured_output/):

```python
class Route(BaseModel):
    """Where a Luche Bank support message should go."""

    team: Literal["cards", "transfers", "loans", "fraud"]
    urgency: int = Field(ge=1, le=5)
    reason: str

router = Agent(model=model, output_type=Route, system_prompt="You triage support messages.")
```

Then dispatch is a dict lookup:

```python
route = router.run_sync(message).output
answer = SPECIALISTS[route.team].run_sync(message)
```

That is the whole pattern. There is no `Router` class in Sophons, deliberately
— a wrapper around two lines would hide the thing worth seeing.

## Why `Literal` matters

`Literal` becomes a JSON schema `enum`, so the model is handed exactly the
valid choices and Pydantic rejects anything else on the way back:

```json
"team": {
  "enum": ["cards", "transfers", "loans", "fraud"],
  "type": "string"
}
```

A route cannot come back as an invalid category, or as prose that happens to
mention a team name. That is what makes `SPECIALISTS[route.team]` safe to
write as a bare lookup rather than a parse-and-hope.

Grading comes free from the same call. Given nothing but
`Field(description="1 routine to 5 critical")`:

```
fraud      u5  Someone withdrew 80,000 from my account and it was not me!
loans      u1  What is the interest rate on a car loan?
cards      u3  My card was swallowed by the ATM at Westlands
```

## Examples

_Nothing here yet — see the roadmap._

## Where this goes next

Routing dispatches to a specialist and stops. When the specialist can hand
control onward, or hand it back, that is multi-agent rather than routing —
same decision, landing on an agent instead of a handler. And a plan is this
decision repeated over an ordered list, which is `planning/`.
