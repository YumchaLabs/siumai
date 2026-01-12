# Ollama Official API Alignment (Chat + Embeddings)

This document records **official API correctness checks** for Ollama’s local HTTP API,
and how Siumai maps/validates the same behavior (during the Alpha.5 fearless refactor).

It complements:

- `docs/alignment/provider-implementation-alignment.md` (global provider audit checklist)

## Sources (Ollama docs)

- Ollama HTTP API: <https://github.com/ollama/ollama/blob/main/docs/api.md>

## Siumai implementation (where to compare)

- Provider: `siumai-provider-ollama/src/providers/ollama/*`
- ProviderSpec: `siumai-provider-ollama/src/providers/ollama/spec.rs`
- Standard mapping (shared helpers): `siumai-provider-ollama/src/standards/ollama/*`

## Base URL + endpoints (official)

Default local server:

- Base URL: `http://localhost:11434`

Endpoints used by Siumai:

- Chat: `POST /api/chat`
- Embeddings: `POST /api/embed`
- Models list: `GET /api/tags`

## Authentication (official)

Ollama typically runs locally and does not require auth.

### Siumai mapping

- No auth headers by default; extra headers are passed through from `ProviderContext.http_extra_headers`.

## Streaming protocol (official)

Ollama chat streaming is JSON Lines: each chunk is a JSON object delimited by `\\n`.

### Siumai mapping

- Streaming ingress/egress uses `JsonEventConverter`:
  - `siumai-provider-ollama/src/providers/ollama/streaming.rs` (`OllamaEventConverter`)

## Tests (how correctness is locked down)

- Chat request fixtures: `siumai/tests/ollama_chat_request_fixtures_alignment_test.rs`
- HTTP errors: `siumai/tests/ollama_http_error_fixtures_alignment_test.rs`

## Status

- Ollama is treated as **Green** for endpoint mapping and fixture parity.
