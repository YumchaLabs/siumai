# Protocol Bridge + Gateway Runtime - Path Audit

Last updated: 2026-03-21

This note records the current explicit bridge paths, where conversion is exact or projected, and
where behavior is still intentionally lossy or only available through adjacent legacy hooks.

## Architectural summary

The current bridge shape is:

1. protocol-native request JSON -> protocol-owned typed normalization -> `ChatRequest`
2. normalized runtime execution
3. normalized `ChatResponse` / V3 stream parts -> target protocol serializer

The repository does not use a runtime `N x M` bridge plugin mesh.

Current explicit request-source normalization entry points:

- `bridge_anthropic_messages_json_to_chat_request(...)`
- `bridge_gemini_generate_content_json_to_chat_request(...)`
- `bridge_openai_responses_json_to_chat_request(...)`
- `bridge_openai_chat_completions_json_to_chat_request(...)`

Current explicit request-source normalization customization entry points:

- `bridge_anthropic_messages_json_to_chat_request_with_options(...)`
- `bridge_gemini_generate_content_json_to_chat_request_with_options(...)`
- `bridge_openai_responses_json_to_chat_request_with_options(...)`
- `bridge_openai_chat_completions_json_to_chat_request_with_options(...)`

Current explicit normalized response targets:

- `bridge_chat_response_to_openai_responses_json_value(...)`
- `bridge_chat_response_to_openai_chat_completions_json_value(...)`
- `bridge_chat_response_to_anthropic_messages_json_value(...)`
- `bridge_chat_response_to_gemini_generate_content_json_value(...)`

Current explicit normalized stream targets:

- `bridge_chat_stream_to_openai_responses_sse(...)`
- `bridge_chat_stream_to_openai_chat_completions_sse(...)`
- `bridge_chat_stream_to_anthropic_messages_sse(...)`
- `bridge_chat_stream_to_gemini_generate_content_sse(...)`

Current curated direct request pair bridges:

- Anthropic Messages -> OpenAI Responses
- OpenAI Responses -> Anthropic Messages

Everything else currently goes through the normalized backbone.

## Exact / projected / lossy / implicit

The terms below are used intentionally:

- `exact`: the bridge preserves the currently modeled semantics for the tested shape
- `projected`: same-protocol replay preserves the stable semantic projection, not raw wire equality
- `lossy`: the bridge is explicit and reported, but some semantics are dropped or collapsed
- `implicit`: behavior still depends on adjacent hooks outside the explicit bridge surface

## Protocol status

| Protocol view | Inbound request path | Outbound response path | Outbound stream path | Current status |
| --- | --- | --- | --- | --- |
| Anthropic Messages | explicit | explicit | explicit | projected / lossy depending on usage + provider metadata |
| OpenAI Responses | explicit | explicit | explicit | strongest exactness today, with a few documented lossy source-replay edges |
| OpenAI Chat Completions | explicit | explicit | explicit | exact for core text/tool/usage/top-level fields, lossy for reasoning/tool-result-only semantics |
| Gemini GenerateContent | explicit | explicit | explicit | request ingress now exists; response fidelity is still intentionally projected/lossy |

## Anthropic Messages

### Inbound

Path:

- Anthropic Messages JSON
- `bridge_anthropic_messages_json_to_chat_request(...)`
- normalized `ChatRequest`

Current fidelity:

- exact / projected for base request settings, function tool choice, provider-defined tools,
  structured output restoration, MCP server options, and thinking settings covered by fixtures
- inbound post-normalize customization now reuses the typed bridge hook/remapper/loss-policy
  surface through the `with_options` normalization entry points
- no whole-parser override trait by design

### Outbound non-streaming response

Path:

- normalized `ChatResponse`
- response inspection + loss reporting
- Anthropic Messages serializer / provider transformer surface

Current fidelity:

- projected exactness for text, tool-call sequence, and supported thinking replay fields
- lossy for `prompt_tokens_details` / `completion_tokens_details`
- lossy for `system_fingerprint`
- lossy for provider metadata outside the Anthropic namespace or outside the currently mapped
  content-part metadata surface
- provider-executed synthetic tool-result replay is not yet always exact on same-protocol
  roundtrip

### Outbound streaming response

Path:

- normalized V3 stream parts
- Anthropic event serializer / finalizer

Current fidelity:

- projected exactness for start metadata, finish reason, prompt/completion totals, and text
  reconstruction
- raw SSE frame-for-frame equality is intentionally not a goal because block framing and terminal
  events may be regenerated from normalized stream state

## OpenAI Responses

### Inbound

Path:

- OpenAI Responses JSON
- `bridge_openai_responses_json_to_chat_request(...)`
- normalized `ChatRequest`

Current fidelity:

- exact / projected for assistant items, tool calls, provider-executed tool items, and most
  request settings covered by fixtures
- best-effort restoration for `item_reference`
- lossy when only `function_call_output` remains and the original tool identity cannot be recovered

### Outbound non-streaming response

Path:

- normalized `ChatResponse`
- response inspection + loss reporting
- OpenAI Responses serializer

Current fidelity:

- exact for core text, reasoning blocks, provider-executed tool results, tool approval requests,
  usage detail breakdown, response metadata, and supported source metadata
- same-protocol roundtrip is now fixture-covered for exact cases, including web-search
  tool-result embedded source replay and file-search typed source projection from raw provider
  results
- one documented non-exact edge remains around normalized tool-scoped provider metadata source
  reconstruction: if user-supplied source linkage is keyed by unified `tool_call_id` but OpenAI
  provider-executed items rebind through `itemId`, `provider_metadata.openai.sources` still
  reflects projected source ids / linkage rather than byte-for-byte restoration

### Outbound streaming response

Path:

- normalized V3 stream parts
- OpenAI Responses event serializer / finalizer

Current fidelity:

- projected exactness for response metadata, finish reason, prompt/completion totals, text
  reconstruction, and reasoning boundary identity
- raw event-for-event equality is intentionally not required

## OpenAI Chat Completions

### Inbound

Path:

- OpenAI Chat Completions JSON
- `bridge_openai_chat_completions_json_to_chat_request(...)`
- normalized `ChatRequest`

Current fidelity:

- exact / projected for system messages, assistant tool calls, tool results, file inputs, and
  response-format JSON schema restoration
- some system-message-mode variants remain best-effort restoration rather than literal wire replay

### Outbound non-streaming response

Path:

- normalized `ChatResponse`
- response inspection + loss reporting
- OpenAI Chat Completions serializer

Current fidelity:

- exact for text, tool calls, usage totals/details, `system_fingerprint`, and `service_tier`
- same-protocol roundtrip is now covered for the normalized projection of those fields
- lossy for reasoning blocks
- lossy for tool-result-only content parts and tool approval request / response semantics
- lossy when stop semantics collapse into the smaller Chat Completions finish-reason space
- lossy for top-level provider metadata outside native Chat Completions fields

### Outbound streaming response

Path:

- normalized V3 stream parts
- OpenAI Chat Completions event serializer / finalizer

Current fidelity:

- projected exactness for final response id / model / timestamp seeding and finish chunk behavior
- cross-protocol best-effort stream routes are explicitly marked lossy so `BridgeMode::Strict`
  rejects them consistently

## Gemini GenerateContent

### Inbound

Path:

- Gemini GenerateContent JSON
- `bridge_gemini_generate_content_json_to_chat_request(...)`
- normalized `ChatRequest`

Current fidelity:

- exact / projected for:
  - system instruction
  - user / model / tool message reconstruction
  - function declarations plus Google provider-defined tools
  - `toolConfig.functionCallingConfig` and retrieval config
  - `cachedContent`, `safetySettings`, labels, and core generation config provider options
  - `responseJsonSchema` restoration through normalized structured output plus provider options
- projected exactness for provider-executed `code_execution` replay:
  - `executableCode`
  - `codeExecutionResult`
- remaining risk is broader fixture coverage, not the lack of an explicit ingress path

### Outbound non-streaming response

Path:

- normalized `ChatResponse`
- response inspection + loss reporting
- Gemini GenerateContent serializer

Current fidelity:

- projected exactness for:
  - visible text
  - tool calls
  - aggregate usage totals
  - `thoughtsTokenCount`
  - `cachedContentTokenCount`
  - `responseId`
  - `modelVersion`
  - preserved `groundingMetadata` / `urlContextMetadata`
  - re-derived source lists when grounding metadata or normalized source parts are present
- lossy for reasoning blocks
- lossy for `system_fingerprint` and `service_tier`
- lossy for prompt/completion audio breakdown and prediction-token breakdown

### Outbound streaming response

Path:

- normalized V3 stream parts
- Gemini event serializer / finalizer

Current fidelity:

- supported as an explicit target stream view
- constrained by the same capability limits as the non-streaming Gemini response view
- raw event equality is not claimed

## Still implicit today

These extension points still exist and remain useful, but they are adjacent to the explicit bridge
surface rather than replacing it:

- `ProviderRequestHooks`
- `ExecutionPolicy::before_send`
- `LanguageModelMiddleware::transform_json_body`
- `LanguageModelMiddleware::post_generate`
- `LanguageModelMiddleware::on_stream_event`
- Axum helper-local transform closures

The current recommendation remains:

- prefer explicit bridge entry points for protocol conversion
- prefer `BridgeOptions`, typed bridge hooks, remappers, and loss policies for customization
- use the adjacent implicit hooks only as escape hatches or for concerns that truly sit outside the
  bridge boundary

## Current conclusion

The current architecture is already coherent:

- request normalization is explicit for Anthropic Messages, OpenAI Responses, and OpenAI Chat
  Completions, and Gemini GenerateContent
- response and stream serialization are explicit for all four protocol views
- a small curated direct-pair layer exists only where it materially reduces request loss
- remaining work is mostly fidelity auditing and broader fixture coverage
