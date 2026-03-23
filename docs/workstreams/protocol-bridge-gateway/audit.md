# Protocol Bridge + Gateway Runtime - Path Audit

Last updated: 2026-03-23

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
| OpenAI Chat Completions | explicit | explicit | explicit | exact/projected for core text/tool/usage fields, structurally lossy for reasoning and tool-result-only semantics |
| Gemini GenerateContent | explicit | explicit | explicit | projected for same-protocol reasoning/tool replay, lossy for finish/detail-breakdown edges |

## Cross-target semantic audit

This workstream's remaining fidelity questions are now concentrated in three semantic families:

- reasoning
- structured output
- usage

The table below records the current target-protocol boundary rather than raw provider-native wire
equality.

| Target protocol view | Reasoning | Structured output | Usage | Primary remaining gaps |
| --- | --- | --- | --- | --- |
| Anthropic Messages | projected exactness for supported thinking replay fields on non-streaming response; stream replay stays stateful/projected rather than block-for-block exact | primarily request-side via `output_format`; outbound response/stream do not add a separate schema envelope beyond normal content replay | prompt/completion totals survive where representable; detailed token breakdown is lossy | `prompt_tokens_details` / `completion_tokens_details`, non-Anthropic provider metadata, broader stream thinking fixtures |
| OpenAI Responses | strongest target today: non-streaming response keeps modeled reasoning blocks exactly, stream keeps reasoning boundary identity semantically | primarily request-side via `text.format`; outbound response does not need a separate schema envelope beyond normalized content/items | non-streaming response keeps usage detail breakdown; stream path is currently audited at prompt/completion totals rather than every nested detail field | message-citation source replay, broader stream-level usage-detail audit |
| OpenAI Chat Completions | structurally lossy by target design because Chat Completions has no first-class reasoning block channel in non-streaming or streaming output | primarily request-side via `response_format`; outbound response/stream are not schema-carrying targets | non-streaming response keeps usage totals/details; stream replay currently guarantees totals and finish semantics, not a richer detail envelope | reasoning loss, tool-result-only semantics, smaller finish-reason space, totals-only stream usage view |
| Gemini GenerateContent | projected exactness for same-protocol visible reasoning partitioning, `thoughtSignature`, and provider-executed `code_execution` replay on response/stream paths | primarily request-side via `responseJsonSchema`; outbound response/stream do not carry a second schema envelope beyond content replay | aggregate totals plus `thoughtsTokenCount` / `cachedContentTokenCount` survive; richer audio/prediction breakdown does not | client tool-call `finishReason` collapse into generic `STOP`, audio/prediction-token breakdown, broader source/usage fixture expansion |

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

- projected exactness for text, provider-hosted tool-call / tool-result replay, tool caller
  metadata, MCP server-name metadata, and supported thinking replay fields
- structured output remains primarily a request-side Anthropic concern (`output_format`) rather
  than a distinct response envelope problem on this target
- lossy for `prompt_tokens_details` / `completion_tokens_details`
- lossy for `system_fingerprint`
- lossy for provider metadata outside the Anthropic namespace or outside the currently mapped
  content-part metadata surface

### Outbound streaming response

Path:

- normalized V3 stream parts
- Anthropic event serializer / finalizer

Current fidelity:

- projected exactness for start metadata, finish reason, prompt/completion totals, and text
  reconstruction
- reasoning/tool replay remains intentionally stateful/projected rather than block-for-block raw
  SSE equality
- structured output is still primarily a request-side concern on this target; no separate stream
  schema envelope is expected
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
- structured output is primarily expressed on the request side (`text.format`), so response-side
  fidelity is about normalized content/items rather than replaying a second schema envelope
- same-protocol roundtrip is now fixture-covered for exact cases, including web-search
  tool-result embedded source replay, file-search typed source projection from raw provider
  results, and tool-scoped source id / linkage preservation even when OpenAI `itemId` differs
  from the unified `tool_call_id`
- remaining non-exact source replay is now mainly limited to message-citation annotations, whose
  source ids may still be regenerated during annotation-based reconstruction

### Outbound streaming response

Path:

- normalized V3 stream parts
- OpenAI Responses event serializer / finalizer

Current fidelity:

- projected exactness for response metadata, finish reason, prompt/completion totals, text
  reconstruction, reasoning boundary identity, and the currently audited provider-hosted tool
  families covered by same-protocol fixtures
- same-protocol stream replay now explicitly preserves:
  - MCP provider tool-call / tool-result replay without degrading calls into generic
    function-call-only semantics
  - finish-carried output-text `logprobs` on OpenAI Responses streams
  - provider-hosted tool families currently fixture-covered in stream roundtrip tests:
    - apply-patch
    - code-interpreter
    - file-search
    - image-generation
    - local-shell
    - MCP tool + approval flows
    - shell
    - web-search
- the stream path is currently audited at semantic totals/boundaries, not every nested usage-detail
  field carried by non-streaming Responses JSON
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
- structured output is still primarily request-side on this target (`response_format`) rather than
  a distinct response envelope surface
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

- projected exactness for assistant role/text replay, tool-call argument accumulation, usage-total
  replay, terminal finish chunk behavior, and preserving the absence of start metadata when the
  source stream did not provide `id` / `model` / `created`
- reasoning remains lossy because Chat Completions has no first-class reasoning stream channel
- structured output remains a request-side concern on this target, not a stream target surface
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
  - reasoning blocks
  - part-level `thoughtSignature`
  - tool calls
  - provider-executed `code_execution` call/result replay
  - aggregate usage totals
  - `thoughtsTokenCount`
  - `cachedContentTokenCount`
  - `responseId`
  - `modelVersion`
  - preserved `groundingMetadata` / `urlContextMetadata`
  - re-derived source lists when grounding metadata or normalized source parts are present
- structured output is primarily request-side on this target via `responseJsonSchema`, not a
  separate response envelope
- lossy for client tool-call `finishReason` collapse into generic `STOP`
- lossy for `system_fingerprint` and `service_tier`
- lossy for prompt/completion audio breakdown and prediction-token breakdown

### Outbound streaming response

Path:

- normalized V3 stream parts
- Gemini event serializer / finalizer

Current fidelity:

- supported as an explicit target stream view under both `google` and `google-vertex` builds
- projected exactness for same-protocol text replay, reasoning replay with provider-namespace
  `thoughtSignature` preservation, provider-executed `code_execution` call/result replay, and
  aggregate usage replay
- the public `google` feature surface now has an explicit protocol-crate integration test entry
  that exercises `GeminiEventConverter` directly for:
  - provider-executed `code_execution` call/result re-serialization
  - reasoning `thoughtSignature` preservation through the paired
    `reasoning-start` / `reasoning-delta` / `ThinkingDelta` path
  - no duplicate standalone reasoning chunk emission before the visible thinking frame
- constrained by the same non-aggregate usage/detail limits as the non-streaming Gemini response
  view
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
- the remaining work is now mostly broader fixture expansion and selective gap closure, not
  uncertainty about the current target-level reasoning / structured-output / usage boundary
