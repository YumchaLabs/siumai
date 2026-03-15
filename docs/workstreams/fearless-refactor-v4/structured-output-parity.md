# Fearless Refactor V4 - Structured Output Parity

Last updated: 2026-03-12

This document tracks the V4 alignment work for **structured output** (schema-driven output)
across providers. It is aligned with the Vercel AI SDK `responseFormat` surface, while
keeping Siumai's Rust-first type naming.

## Stable surface (canonical)

The Stable surface is `ChatRequest.response_format: Option<ResponseFormat>`.

Current Stable type:

- `ResponseFormat::Json { schema, name, description, strict }`

Stable helper (best-effort extraction):

- `siumai::structured_output::extract_json_value_from_response(&ChatResponse)`
- `siumai::structured_output::extract_json_from_response::<T>(&ChatResponse)`
- `siumai::structured_output::extract_json_value_from_stream(TextStream)`
- `siumai::structured_output::extract_json_from_stream::<T>(TextStream)`

Canonical public example:

- `siumai/examples/03-advanced-features/provider-params/structured-output.rs`
  now demonstrates the Stable request + typed extraction path directly, instead of
  routing the example through provider-specific OpenAI request options.

Stable failure semantics:

- Non-streaming / explicit `StreamEnd` responses remain best-effort and may still repair minor JSON issues
  such as fenced output, trailing commas, or lightly malformed object syntax.
- Streaming extraction without an explicit `StreamEnd` now requires a complete JSON value; truncated
  partial output no longer silently succeeds through JSON repair.
- That same interrupted-stream rule now also applies to reserved `json` tool payloads: if a stream
  stops before the tool arguments form a complete JSON value, extraction returns the same dedicated
  incomplete-stream parse error instead of leaking a generic parser-specific message.
- If the final response ends with `FinishReason::ContentFilter` and no valid JSON payload was produced,
  extraction returns a parse error that explicitly reports content filtering / refusal instead of a
  generic "no JSON candidate found" failure.
- Typed helpers now also distinguish the second failure stage explicitly: if JSON extraction succeeded
  but deserialization into the target Rust type failed, the error message reports a target-type mismatch
  instead of reusing the generic "failed to parse JSON response" wording.
- The public `siumai::structured_output` facade now delegates typed extraction to `siumai-core`
  directly, and integration coverage pins that the facade preserves both typed-mismatch errors and
  strict interrupted-stream behavior instead of reintroducing a second parsing path.
- Public facade coverage now also pins the two remaining high-value response boundaries directly:
  `FinishReason::ContentFilter` yields the dedicated refusal/content-filter parse error, and
  explicit `StreamEnd` responses still keep the best-effort repair path even if earlier deltas
  were incomplete.
- Streams that terminate through a synthetic `StreamEnd` with `FinishReason::Unknown` now also
  reuse accumulated deltas / reserved-tool arguments before deciding success vs incomplete-stream
  failure, so JSON-streaming providers such as Ollama no longer lose already-emitted structured
  output when the transport closes after deltas but before a provider-native terminal frame.

Canonical meaning:

- The caller requests that the model outputs JSON conforming to the given JSON schema.
- `name` optionally identifies the schema (default: `"response"`).
- `description` optionally describes the expected JSON output (provider-dependent support).
- `strict` optionally requests strict schema adherence (provider-dependent support).
- Providers may use native schema features when available, otherwise they may fall back
  to tool-based strategies (reserved tools) to obtain structured output.

## Provider mappings (current)

### OpenAI (Chat Completions)

Maps to `response_format` with `json_schema`:

- `{ type: "json_schema", json_schema: { name, schema, strict, description? } }`
- `name` defaults to `"response"`.
- `description` is included only when non-empty.
- `strict` resolution: `ResponseFormat.strict` > `providerOptions.openai.strictJsonSchema` > default `true`.

### OpenAI (Responses API)

Maps to `text.format`:

- `{ type: "json_schema", name, schema, strict }`

### Groq

Uses the OpenAI Chat Completions mapping via the Groq adapter:

- `response_format = { type: "json_schema", json_schema: { name, schema, strict } }`
- Groq-specific request adaptation keeps `response_format` intact while only normalizing roles,
  `stream_options`, and `max_completion_tokens -> max_tokens`
- Priority rules:
  - Stable `ChatRequest.response_format` wins over raw `providerOptions.groq.response_format` when both are present.
  - Groq typed provider options for `logprobs`, `top_logprobs`, `service_tier`, and reasoning hints still merge alongside the Stable structured-output request.

### xAI

Uses the OpenAI-compatible runtime-provider path:

- `response_format = { type: "json_schema", json_schema: { name, schema, strict } }`
- Runtime provider id keyed routing (`xai`) preserves both `response_format` and `tool_choice` semantics
- Priority rules:
  - Stable `ChatRequest.response_format` wins over raw `providerOptions.xai.response_format` when both are present.
  - xAI request normalization still strips unsupported `stop` / `stream_options` fields after provider-option merging.
  - xAI-specific typed request options for `reasoningEffort` / `searchParameters` still normalize onto the final snake_case wire shape alongside Stable structured output.

### Anthropic

Two strategies:

1. Models that support native `output_format`:
   - `output_format: { type: "json_schema", schema }`
   - `name` / `description` / `strict` are currently ignored.
2. Fallback (reserved tool strategy):
   - Inject a reserved `json` tool with `input_schema`
   - Force `tool_choice: { type: "any", disable_parallel_tool_use: true }`
   - Non-streaming and streaming both normalize the reserved `json` tool output into plain JSON text on the Stable surface.

### Gemini

Maps to `generationConfig`:

- `generationConfig.responseMimeType = "application/json"`
- `generationConfig.responseSchema = <schema>`
- `name` / `description` / `strict` are currently ignored.
- Priority rules:
  - `ResponseFormat::Json` always forces `generationConfig.responseMimeType = "application/json"`, even if provider options supplied another MIME type.
  - `providerOptions.google.structuredOutputs = true` (default behavior) prefers `generationConfig.responseSchema` and removes legacy `generationConfig.responseJsonSchema` when both are present.
  - `providerOptions.google.structuredOutputs = false` keeps `responseFormat`'s JSON MIME type but skips `responseSchema`; if `providerOptions.google.responseJsonSchema` was supplied, that legacy field remains in the final request.

### Amazon Bedrock (Converse)

Uses a reserved tool strategy (Bedrock does not expose an OpenAI-like `response_format` field):

- Inject a reserved `json` function tool with `inputSchema.json = <schema>`.
- Force `toolConfig.toolChoice = { any: {} }`.
- When the model responds with a `toolUse` for `"json"`, the `input` payload is stringified into
  a text part so the Stable surface receives plain JSON.
- Public-path parity now also locks the streaming reserved-tool boundary directly: if the Converse
  JSON stream closes after reserved `json` tool deltas but before a terminal `messageStop`, complete
  accumulated tool input still parses while truncated tool input returns the dedicated incomplete-stream
  parse error across builder/provider/config construction.

### Ollama

Maps to `/api/chat` `format`:

- `format = <schema>` when `ChatRequest.response_format = ResponseFormat::Json { .. }`
- Request-level `response_format` overrides `providerOptions["ollama"].format`
- Response parsing remains text-first; Stable JSON extraction uses the shared structured-output helper

## Acceptance tests (no network)

The mapping is considered "shipped enough" when:

1. Request transformers lock the wire shape for each provider.
2. Provider specs do not accidentally override request-level intent.
3. One test asserts the main override knob (`strictJsonSchema`) where applicable.

Current status:

- OpenAI Chat Completions: request transformer maps `response_format` to `json_schema` wire shape.
- OpenRouter: public alignment tests now also lock that raw `providerOptions.openrouter.response_format` does not override Stable `response_format`, while OpenRouter-specific vendor params such as `transforms` still merge into the final request body. The same path now also has typed-request coverage through `OpenRouterOptions`, proving typed vendor params can coexist with Stable structured output. Vendor-view public-path parity now also locks structured-output extraction on the shared SSE route when the stream closes without a provider-native terminal finish frame: complete accumulated JSON still parses, while truncated output returns the dedicated incomplete-stream parse error across builder/provider/config/registry construction.
- Perplexity: public alignment tests now also lock that raw `providerOptions.perplexity.response_format` does not override Stable `response_format`, while generic vendor params still merge into the final request body on the shared OpenAI-compatible path. Typed `PerplexityOptions` coverage now locks the same coexistence for common hosted-search fields.
- Perplexity: vendor-view public-path parity now also locks structured-output extraction on the shared SSE route when the stream closes without a provider-native terminal finish frame: complete accumulated JSON still parses, while truncated output returns the dedicated incomplete-stream parse error across builder/provider/config/registry construction.
- Groq: provider-level adapter smoke tests confirm `response_format` survives wrapper-specific request normalization, and no-network spec tests now also lock that Stable `response_format` is not overridden by raw Groq provider options while typed logprobs/reasoning knobs still merge.
- xAI: runtime-provider smoke tests confirm OpenAI-compatible `response_format` and `tool_choice` semantics survive provider-id keyed routing, and no-network coverage now also locks Stable `response_format` precedence over raw xAI provider options plus post-merge stripping of unsupported `stop` / `stream_options` fields. The generic `OpenAiCompatibleClient` path now also has transport-boundary capture tests for both non-streaming and streaming requests, so the final xAI wire body is guarded even without the provider-owned wrapper. Public-path parity now also locks structured-output extraction on the shared SSE route when the stream closes without a provider-native terminal finish frame: complete accumulated JSON still parses, while truncated output returns the dedicated incomplete-stream parse error across builder/provider/config/registry construction.
- DeepSeek: runtime-provider smoke tests and provider-owned request-capture coverage now confirm Stable `response_format` survives both provider-id keyed routing and `DeepSeekSpec` normalization, so raw `providerOptions.deepseek.response_format` no longer overrides the Stable request while `reasoningBudget` still normalizes onto the final snake_case wire shape. The shared OpenAI-compatible runtime now also normalizes `enableReasoning` / `reasoningBudget` on the generic `OpenAiCompatibleClient` path, and a direct transport-boundary test guards the final runtime-provider request body there as well.
- Anthropic: request transformer uses `output_format` when supported, otherwise injects reserved `json` tool strategy; response and streaming paths both collapse the reserved `json` tool into plain JSON text. Public-path parity now also locks the forced `jsonTool` fallback on the real Anthropic SSE route: when reserved `json` tool deltas are already complete and the stream closes before `message_stop`, extraction still succeeds across builder/provider/config/registry construction, while truncated tool input returns the dedicated incomplete-stream parse error. The provider-owned escape hatch is now typed as `provider_ext::anthropic::AnthropicStructuredOutputMode`, wired through `AnthropicOptions::with_structured_output_mode(...)` instead of forcing raw `with_provider_option(...)` calls.
- Gemini: request transformer now also locks precedence between `responseFormat`, `structuredOutputs`, and legacy `responseJsonSchema`, including MIME override semantics and the schema-field switch between `responseSchema` and `responseJsonSchema`.
- Ollama: request transformer maps Stable `response_format` to `/api/chat` `format`, with request-level intent overriding provider options.

## Known gaps / next steps

- Revisit whether deserialization helpers should expose a richer typed error boundary than
  `LlmError::ParseError` when callers need enum-level branching instead of message-level distinction.




