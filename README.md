# llmtrace

A transparent tracing proxy for OpenAI- and Anthropic-compatible LLM gateways.
Put it in front of [FreeLLMAPI](https://github.com/tashfeenahmed/freellmapi) (or any
`/v1` gateway) and every call becomes an OpenTelemetry span in Langfuse, with
**TTFB, TTFT, decode throughput, token usage, finish reason, provider headers and
errors**, without buffering or altering the stream.

```
clients (scripts, Claude Code, Codex, agents)
        │  OPENAI_BASE_URL=http://llmtrace:8080/v1
        ▼
   llmtrace ──── OTLP/HTTP JSON ────► Langfuse
        │
        ▼
   FreeLLMAPI ──► Groq / Cerebras / Gemini / Mistral / ...
```

- **Zero dependencies.** Standard library only, ships a ~6 MB static binary. The
  OTLP exporter is ~400 lines, sits behind an `Exporter` interface, and batches
  spans off the request path through a bounded queue (it drops spans rather than
  slowing requests down).
- **Streaming-safe.** `FlushInterval: -1` plus a tee reader means chunks reach the
  client the moment upstream sends them. A test enforces this.
- **Understands three wire formats:** OpenAI Chat Completions (plus legacy
  completions), the OpenAI Responses API, and Anthropic Messages, both streamed
  and non-streamed.
- **Fixes missing stream usage.** Injects `stream_options.include_usage=true` on
  OpenAI-style streams so providers report token counts.
- **W3C trace context.** It honours an incoming `traceparent` (your agent's span
  becomes the parent) and forwards its own span upstream.

## What lands in Langfuse

| Langfuse field | Source |
|---|---|
| Observation type | `generation` (`embedding` for `/embeddings`) |
| Model | the response's `model`, falling back to the request's. With `model: "auto"` you see what actually served it. |
| Usage | `usage_details` `{input, output, total, input_cache_read?}` |
| Completion start time | first content/reasoning/tool-call delta, which gives TTFT in the Langfuse UI |
| Provider | `gen_ai.provider.name` from the response's `provider` field (OpenRouter-style gateways), e.g. `AtlasCloud` |
| Stream shape | `llm.stream.chunks`, `generation_window_ms`, `max_gap_ms`, `median_gap_ms`, and `stream_buffered`. A stream that spends more than half its time before the first token and then delivers 10+ chunks back to back (median gap under 1 ms) is flagged `WARNING`: upstream buffered the generation and replayed it, so TTFT ≈ total latency. |
| Filterable metadata | `ttft_ms`, `ttfb_ms`, `output_tokens_per_sec`, `http_status`, `stream`, `endpoint`, `upstream.x-*` headers |
| Level / status | `ERROR` with the upstream error body on 4xx/5xx/timeouts (429s become visible) |
| Session / user | `X-Session-Id` / `X-User-Id` headers (or `X-Langfuse-*`), the OpenAI `user` field, Anthropic `metadata.user_id` |
| Input / output | only when `CAPTURE_CONTENT=true` (capped by `MAX_CAPTURE_BYTES`) |

Every request also logs one JSON line (`msg=llm_request`) with the same numbers.
That's the raw feed for aggregate dashboards later.

## Configuration

All configuration is through environment variables. See `.env.example`.

| Var | Default | |
|---|---|---|
| `UPSTREAM_URL` | required | e.g. `http://freellmapi:3001` |
| `LISTEN_ADDR` | `:8080` | |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | unset = logs only | Langfuse: `<host>/api/public/otel` |
| `OTEL_EXPORTER_OTLP_HEADERS` | | `Authorization=Basic%20<b64(pk:sk)>,x-langfuse-ingestion-version=4` |
| `CAPTURE_CONTENT` | `false` | send prompts/completions |
| `MAX_CAPTURE_BYTES` | `32768` | |
| `INJECT_INCLUDE_USAGE` | `true` | |
| `FORWARD_TRACEPARENT` | `true` | |
| `LLMTRACE_ENVIRONMENT` | | becomes `langfuse.environment` |

## Dev loop

The code runs on your laptop. Only the deployed copy lives on the VM.

```bash
go test -race ./...                              # unit tests with fake upstreams
UPSTREAM_URL=https://<your-freellmapi-domain> go run ./cmd/llmtrace
curl -N localhost:8080/v1/chat/completions -H "Authorization: Bearer $FREELLMAPI_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"auto","stream":true,"messages":[{"role":"user","content":"hi"}]}'
```

Point it at FreeLLMAPI's public URL while developing. In production it talks
over the internal Docker network.

## Deploy on Coolify

1. **Push this repo to GitHub** (`Zoronium/llmtrace`).
2. **Langfuse:** in Coolify, add a new resource, choose the Langfuse one-click
   service, and deploy it. Create a project in the Langfuse UI and copy its
   public and secret keys. Then run `echo -n "pk-lf-...:sk-lf-..." | base64 -w 0`.
3. **llmtrace:** add a new resource from your GitHub repo, set the build pack to
   **Dockerfile**, and set the port to `8080`. Add the env vars from
   `.env.example`, and give it a domain (e.g. `llm.advaitjain.com`).
4. **Wire the network.** On the VM, run:
   ```bash
   docker ps --format '{{.Names}}\t{{.Networks}}' | grep -Ei 'freellm|langfuse|llmtrace'
   ```
   Containers on the same network (Coolify's default is `coolify`) reach each
   other by container name. Use those names for `UPSTREAM_URL` and the Langfuse
   endpoint. If a one-click service sits on its own network, enable
   **Connect to Predefined Network** on it.
5. **Lock down FreeLLMAPI.** Remove its public domain in Coolify so the only way
   in is through llmtrace. Otherwise untraced traffic slips past.
6. **Repoint clients** to `https://llm.advaitjain.com/v1` (the auth header passes
   through unchanged to FreeLLMAPI).
   - OpenAI SDKs: `OPENAI_BASE_URL=https://llm.advaitjain.com/v1`
   - Anthropic / Claude Code: `ANTHROPIC_BASE_URL=https://llm.advaitjain.com`

## First thing to check

Does FreeLLMAPI say which provider served an `auto` request? After a few calls,
open a trace in Langfuse and look at `gen_ai.response.model` and the
`upstream.x-*` metadata. If the provider isn't there, attribution needs either
pinned models or a small upstream patch to FreeLLMAPI. That patch is a good
first PR.

## Roadmap

- [ ] Synthetic prober: golden prompts every 15 minutes per pinned model, with
      correctness pushed as Langfuse scores (detects silent quantization or model swaps)
- [ ] Aggregates in Postgres feeding a public `status.advaitjain.com` leaderboard
      (p50/p95 TTFT, tok/s, 429 rate per provider)
- [ ] Paid-equivalent cost via `cost_details`
- [ ] Upstream PR: native OTel in FreeLLMAPI
