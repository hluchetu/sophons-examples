#!/usr/bin/env bash
set -e

if ! docker inspect sophons-jaeger >/dev/null 2>&1; then
  docker run -d --rm \
    --name sophons-jaeger \
    -p 16686:16686 \
    -p 4318:4318 \
    cr.jaegertracing.io/jaegertracing/jaeger:2.19.0
fi

echo "Waiting for Jaeger..."

until curl -fsS http://localhost:16686 >/dev/null; do
  sleep 1
done

export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318

uv run observability/traced_agent.py

echo
echo "Open Jaeger: http://localhost:16686"
