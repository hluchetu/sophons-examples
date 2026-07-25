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

### [rag.py](rag.py)

A full RAG evaluation pass: load the Luche Bank docs, retrieve evidence,
generate a grounded answer, then score the completed run across the four
basic RAG dimensions:

```
context_relevance   did retrieval find useful evidence?
faithfulness        is the answer grounded in that evidence?
answer_relevance    did the answer address the question?
answer_correctness  did it match the reference answer?
```

Unlike corrective RAG, this evaluator is not part of the runtime loop.
The RAG system runs first; evaluation measures the finished run.

```bash
uv run evaluation/rag.py
```

### [pass_k.py](pass_k.py)

An `EvalRunner` sweep: a support agent with an order-lookup tool, run over
a versioned test set three times per case, judged on trajectory, tool
parameters, and output. The report shows the two rates that matter:

```
pass rate (per trial):  100%
pass^3 (per case):    100%

dimension averages:
  correctness     1.00
  relevancy       1.00
  tool_parameters 1.00
  trajectory      1.00
```

pass^k (Sierra's tau-bench metric) only credits a case if **every** trial
passes — consistency, not luck.

```bash
uv run evaluation/pass_k.py
```
