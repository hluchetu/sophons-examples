# Observability

Examples of tracing sophons agents with OpenTelemetry.

Sophons components (the agent loop, retrievers, memory, loaders) create spans
unconditionally through the OpenTelemetry **API**, which costs nothing and
records nothing until an application registers an **SDK** tracer provider.
`SophonsTelemetry` is that switch — everything here is variations on flipping
it.

## Examples

### [traced_agent.py](traced_agent.py)

A tool-using agent with the console exporter. Run it and read the span tree
of a single request:

```
invoke_agent                 ← the run: session id, steps, stop reason
├── chat                     ← model call, step 1 (token usage)
├── execute_tool add         ← tool name, call id, step
├── chat                     ← model call, step 2
├── execute_tool multiply
└── chat                     ← final answer
```

```bash
uv run observability/traced_agent.py
```

To see the same run as a waterfall in Jaeger:

```bash
docker run --rm -p 16686:16686 -p 4318:4318 jaegertracing/all-in-one
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
uv run observability/traced_agent.py
# open http://localhost:16686 and search for service "sophons"
```
