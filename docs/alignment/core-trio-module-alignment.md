# Core Trio Module Alignment (Vercel AI SDK + Official APIs)

This document is the **module-level alignment playbook** for the Alpha.5 fearless refactor.
It complements:

- `docs/alignment/m1-core-trio-checklist.md` (what is already covered by tests)
- `docs/alignment/provider-implementation-alignment.md` (global audit checklist across providers)

The goal here is pragmatic:

1. **Map modules** (Vercel file ? Siumai file)
2. Record **what to verify** (official wire protocols + Vercel behavior)
3. Point to **fixtures/tests** that lock the behavior down
4. Track **remaining gaps** as concrete follow-ups

Conventions:

- “Vercel ref” paths are under `repo-ref/ai/packages/*`.
- “Siumai impl” paths are under this repo.
- “Tests” refer to Rust tests/fixtures in `siumai/tests`.

## OpenAI (Chat Completions + Responses)

### Modules map

| Area | Vercel ref | Siumai impl | Tests / fixtures |
| --- | --- | --- | --- |
| Responses request input mapping | `packages/openai/src/responses/convert-to-openai-responses-input.ts` | `siumai-protocol-openai/src/standards/openai/transformers/request.rs` | `siumai/tests/openai_responses_*_fixtures_alignment_test.rs`, `siumai/tests/fixtures/openai/responses/*` |
| Responses hosted tools mapping | `packages/openai/src/responses/openai-responses-prepare-tools.ts`, `packages/openai/src/tool/*` | `siumai-core/src/tools.rs`, `siumai-core/src/hosted_tools/openai.rs`, `siumai-protocol-openai/src/standards/openai/utils.rs` | `siumai/tests/openai_responses_*_fixtures_alignment_test.rs`, `siumai/tests/fixtures/openai/responses/*` |
| Responses SSE parsing | `packages/openai/src/responses/openai-responses-language-model.ts` (`doStream`) | `siumai-protocol-openai/src/standards/openai/responses_sse.rs` | `siumai/tests/openai_responses_*_stream_alignment_test.rs`, `siumai/tests/fixtures/openai/responses-stream/*` |
| Responses SSE serialization (gateway) | Vercel stream parts are internal to `@ai-sdk/provider` | `siumai-protocol-openai/src/standards/openai/responses_sse.rs` (`serialize_event`) | `siumai/tests/transcoding_*_to_openai_alignment_test.rs`, `docs/alignment/streaming-bridge-alignment.md` |
| Error mapping | `packages/openai/src/openai-error.ts` + `packages/provider-utils/src/response-handler.ts` | `siumai-protocol-openai/src/standards/openai/errors.rs` | `siumai/tests/*error*_alignment_test.rs` |
| Network errors retryable | `packages/provider-utils/src/handle-fetch-error.ts` | `siumai-core/src/retry_api.rs` + transports | (covered implicitly by retry tests; add explicit fixture if needed) |

### What to verify (OpenAI)

- **Tool mapping parity**: hosted tool ids ? tool types, and custom tool name mapping:
  - `openai.web_search`, `openai.web_search_preview`
  - `openai.file_search`
  - `openai.code_interpreter`
  - `openai.image_generation`
  - `openai.apply_patch`
  - `openai.local_shell`, `openai.shell`
  - `openai.mcp` (`allowed_tools`, `require_approval`, server fields, approval request id ? dummy toolCallId mapping)
- **Streaming protocol parity (Responses SSE)**:
  - output items (`response.output_item.added/done`)
  - tool deltas (`response.function_call_arguments.delta/done`)
  - sources (`response.output_text.annotation.added`)
  - end-of-stream completion backfill (`response.completed`)
- **Retry / errors**:
  - empty/non-JSON bodies should fall back to HTTP reason phrase (Vercel provider-utils behavior)
  - OpenAI-compatible error code may be a number (Vercel `openai-error.ts` schema)

## Anthropic (Messages)

### Modules map

| Area | Vercel ref | Siumai impl | Tests / fixtures |
| --- | --- | --- | --- |
| Messages request mapping | `packages/anthropic/src/convert-to-anthropic-messages-prompt.ts` | `siumai-protocol-anthropic/src/standards/anthropic/transformers.rs`, `siumai-protocol-anthropic/src/standards/anthropic/utils.rs` | `siumai/tests/anthropic_messages_fixtures_alignment_test.rs`, `siumai/tests/fixtures/anthropic/messages/*` |
| Messages SSE parsing | `packages/anthropic/src/anthropic-messages-language-model.ts` (`doStream`) | `siumai-protocol-anthropic/src/standards/anthropic/streaming.rs` | `siumai/tests/anthropic_messages_stream_fixtures_alignment_test.rs`, `siumai/tests/fixtures/anthropic/messages-stream/*` |
| Tools prepare | `packages/anthropic/src/anthropic-prepare-tools.ts` | `siumai-protocol-anthropic/src/standards/anthropic/utils.rs` (tool mapping helpers) | Anthropic fixtures covering tool search / web search / web fetch / code execution / MCP |
| Error mapping | `packages/anthropic/src/anthropic-error.ts` + `packages/provider-utils/src/response-handler.ts` | `siumai-protocol-anthropic/src/standards/anthropic/errors.rs` | `siumai/tests/*error*_alignment_test.rs` |

### What to verify (Anthropic)

- **Beta headers** for feature-gated tools and endpoints (Vercel tests cover many; keep them stable).
- **Beta header merging rules**: merge config + request `anthropic-beta` with auto-injected betas (lowercased, trimmed, de-duped).
- **Tool definition extensions** (function tools):
  - `defer_loading`, `allowed_callers`, `input_examples` (providerOptions-driven)
  - `strict` only when structured outputs are supported
  - tool caching controls (when supported by the API surface)
- **Streaming tool boundaries**: `tool_use` / `tool_result` semantics and id stability.
- **Retry / errors**: `overloaded_error` is treated as retryable (Vercel-aligned synthetic 529 in Siumai).

## Gemini (Google Generative AI: GenerateContent)

### Modules map

| Area | Vercel ref | Siumai impl | Tests / fixtures |
| --- | --- | --- | --- |
| Request mapping | `packages/google/src/convert-to-google-generative-ai-messages.ts` | `siumai-protocol-gemini/src/standards/gemini/transformers.rs`, `siumai-protocol-gemini/src/standards/gemini/convert.rs` | `siumai/tests/google_generative_ai_fixtures_alignment_test.rs`, `siumai/tests/fixtures/google/generative-ai/*` |
| Streaming parsing + finish mapping | `packages/google/src/google-generative-ai-language-model.ts` (`doStream`) | `siumai-protocol-gemini/src/standards/gemini/streaming.rs` | `siumai/tests/google_generative_ai_stream_fixtures_alignment_test.rs`, `siumai/tests/fixtures/gemini/*` |
| Tools prepare / toolConfig | `packages/google/src/google-prepare-tools.ts` | `siumai-protocol-gemini/src/standards/gemini/convert.rs` (tool mapping + toolChoice), `siumai-protocol-gemini/src/standards/gemini/transformers.rs` | Google fixtures covering toolConfig modes + warnings |
| Official schema reference | (OpenAPI derived) | `docs/gemini_OPENAPI3_0.json` | (doc-driven; keep request transformer tolerant) |

### What to verify (Gemini)

- **Streaming semantics**:
  - `functionCall` parts are tool calls in the stream.
  - `functionResponse` is naturally *next-request input* in Gemini; for gateways we support replay as a stream frame (opt-in).
- **Finish reason mapping**:
  - `STOP` should be `tool-calls` when the chunk includes `functionCall`, otherwise `stop`.
  - Safety-related finish reasons map to `content-filter`.
  - `MALFORMED_FUNCTION_CALL` maps to `error`.
- **ToolConfig parity**: `AUTO` / `ANY` / `NONE` mapping + warnings when mixed unsupported tools are present.
- **Thought signatures**: appear on `functionCall` but not on `functionResponse` (Vercel tests assert this).

## Cross-protocol gateway work (core trio)

This is where module boundaries matter most:

- Bridge parts: `siumai-core/src/streaming/stream_part.rs`, `siumai-core/src/streaming/bridge.rs`
- Transcoding policy: `siumai::experimental::streaming::V3UnsupportedPartBehavior`
- Multi-target SSE gateway helpers: `siumai-extras/src/server/axum.rs`

Recommended workflow for new parity items:

1. Add a fixture (prefer Vercel fixture port) or a minimal mock-api test.
2. Lock the behavior with a Rust test in `siumai/tests`.
3. Update `docs/alignment/vercel-ai-fixtures-alignment.md` and this doc.
