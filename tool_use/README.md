# Tool Use

The function-calling loop with nothing on top of it — one tool, one call,
no memory, no branching. Every other example in this repo assumes this
loop.

## Examples

### [basic.py](basic.py)

The model is asked for a balance it cannot know. It picks the tool, fills
in the argument from the question, and answers from what came back:

```
You    How much is left on account 0114455?

Tool   account_balance({'account_number': '0114455'})
       ->[success] {"result": "KES 12,480.00 available, KES 500.00 on hold"}

Agent  Here's the balance for account 0114455:
         • Available balance: KES 12,480.00
         • Amount on hold: KES 500.00

       steps = 2  model_calls = 2  tool_calls = 1  stop=end_turn
```

Two model calls for one question: the first decides to call the tool, the
second turns the tool's result into a sentence. That round trip is the
whole pattern.

Three things do the work, and all three are read off a plain Python
function by `@tool`:

- the **docstring** becomes the description the model reads when deciding
  whether to call the tool
- the **type hints** become the argument schema it fills in
- the **return value** is wrapped into `{"result": ...}` — which is why
  the tool result above is JSON, not the bare string the function returned

There is no separate schema to maintain.

```bash
uv run tool_use/basic.py
```
