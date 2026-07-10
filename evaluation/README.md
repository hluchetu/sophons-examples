# Evaluation

Examples of judging agent output with `sophons.evals` — evaluation turns
runs into verdicts: a score, a pass/fail, and a reason.

## Examples

### [faithfulness.py](faithfulness.py)

The decompose-then-verify pattern catching a hallucination. Two answers to
the same question are judged against the same retrieved context (which says
nothing about the topic):

```
--- hallucinated
passed=False  score=0.00
reason: Claim 1 is unsupported: context only mentions a 14-day refund
window, not priority support. Claim 2 ... Claim 3 ...

--- honest
passed=True  score=1.00
reason: Answer asserts no factual claims.
```

The invented answer is dismantled claim by claim; the honest "the docs
don't say" passes, because an answer that asserts nothing cannot be
unfaithful.

```bash
uv run evaluation/faithfulness.py
```
