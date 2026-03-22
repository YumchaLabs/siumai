# Protocol Bridge + Gateway Runtime - TODO

Last updated: 2026-03-22

This TODO list is intentionally organized as mergeable tracks.

## 0) Lock the boundary

- [x] Confirm the workstream scope:
  - explicit protocol bridges
  - gateway runtime policy
  - no multi-tenant billing or admin control plane
- [x] Confirm package boundaries:
  - protocol crates own wire formats and serializer/parser state machines
  - `siumai-core` owns bridge contracts and reports
  - `siumai-extras` owns gateway adapters and runtime policy
- [x] Finish the internal bridge module layout:
  - planner
  - request / response / stream
  - primitives
  - pair bridges

## 1) Audit the current bridge path

- [x] Document the current inbound and outbound paths for:
  - Anthropic Messages
  - OpenAI Responses
  - OpenAI Chat Completions
  - Gemini GenerateContent
- [x] Mark which conversions are already exact, lossy, or implicit
  - see `docs/workstreams/protocol-bridge-gateway/audit.md`
- [x] Inventory current customization points:
  - request-side:
    - `RequestTransformer`
    - `ProviderRequestHooks`
    - `ExecutionPolicy::before_send`
    - `LanguageModelMiddleware::transform_json_body`
  - response-side:
    - `to_transcoded_json_response_with_transform`
    - `to_transcoded_json_response_with_response_transform`
    - `LanguageModelMiddleware::post_generate`
  - stream-side:
    - `to_transcoded_sse_response_with_transform`
    - `LanguageModelMiddleware::on_stream_event`
  - gap:
    - none of these expose `BridgeReport`, direct-pair context, or typed bridge lifecycle context
- [x] Identify stateful stream converters that must become explicit bridge dependencies:
  - `OpenAiResponsesStreamPartsBridge`
  - `OpenAiResponsesEventConverter`
  - `OpenAiCompatibleEventConverter`
  - `AnthropicEventConverter`
  - `GeminiEventConverter`

## 2) Define bridge contracts

- [x] Add bridge contract types:
  - `BridgeTarget`
  - `BridgeMode`
  - `BridgeReport`
  - `BridgeWarning`
  - `BridgeResult<T>`
- [x] Define a stable representation for lossy conversion reasons
- [x] Define a stable representation for unsupported semantics
- [x] Decide how provider metadata is carried across bridges
- [x] Add stable bridge customization contract types:
  - `BridgeOptions`
  - `BridgeCustomization`
  - request / response / stream context types
  - primitive remapper context
  - lossy handling policy trait
- [x] Move closure-friendly bridge adapters into a framework-neutral `siumai-extras::bridge`
  module and cover request hook wrappers explicitly

## 3) Make request bridges explicit

- [x] Add explicit normalized request -> target protocol JSON bridge entry points:
  - normalized -> Anthropic Messages
  - normalized -> OpenAI Responses
  - normalized -> OpenAI Chat Completions
  - normalized -> Gemini GenerateContent
- [x] Ensure request bridges can emit `BridgeReport`
- [x] Ensure request bridges can reject unsupported shapes in `Strict` mode
- [x] Refactor request bridge implementation into:
  - planner
  - reusable primitives
  - pair bridge modules
- [x] Add explicit protocol-source typed request -> normalized request bridge entry points for:
  - Anthropic Messages -> normalized request
  - Gemini GenerateContent -> normalized request
  - OpenAI Responses -> normalized request
  - OpenAI Chat Completions -> normalized request
- [x] Add direct compat helpers where they materially reduce loss:
  - Anthropic Messages -> OpenAI Responses
  - OpenAI Responses -> Anthropic Messages

## 3a) Avoid N x M glue code

- [x] Add a request bridge planner that can choose:
  - direct pair bridge
  - via normalized bridge
  - rejected
- [x] Keep direct bridges limited to high-value / high-loss pairs
- [x] Move reasoning / tools / cache control / approval logic into reusable primitives
- [x] Ensure pair bridges compose primitives instead of embedding all mapping logic inline
- [x] Extract reusable hosted-tool translation rule tables from direct pair request bridges
- [x] Isolate OpenAI MCP -> Anthropic `mcp_servers` conversion helpers from pair-specific glue

## 4) Make non-streaming response bridges explicit

- [x] Add explicit normalized response -> target protocol converters
- [x] Emit loss reports for dropped or downgraded semantics
- [x] Add no-network tests for exact and lossy cases
- [x] Preserve tool calls, provider-executed tool results, and OpenAI response sources where
  representable
- [ ] Audit remaining reasoning / structured output / usage fidelity gaps per target
  - OpenAI Responses same-protocol response roundtrip now has initial fixture coverage with:
    - exact cases for core text / reasoning / tool items, including web-search embedded
      tool-result source replay, file-search source extraction from raw provider results, and
      tool-scoped source id / linkage preservation across distinct OpenAI `itemId` and unified
      `tool_call_id`
    - remaining non-exact source behavior is now mainly message-citation annotation replay
  - OpenAI Chat Completions response bridge now has explicit contract coverage for:
    - preserved native top-level fields (`system_fingerprint`, `service_tier`, OpenAI usage totals/details)
    - documented lossy / rejected cases for reasoning blocks, tool results, tool approval requests,
      stop-sequence finish reasons, and top-level provider metadata
    - initial same-protocol response fixture roundtrip coverage for:
      - legacy `function_call` replay
      - assistant text + tool-call replay
      - usage breakdown replay
      - preserved top-level chat-completions fields (`system_fingerprint`, `service_tier`)
  - Anthropic Messages same-protocol response roundtrip now has initial projected fixture
    coverage with:
    - exact/projected JSON output cases
    - exact projected replay for provider-hosted tool-search tool-call / tool-result pairs,
      including raw server tool names and caller metadata when carried on tool-call metadata
    - carried MCP server-name metadata on provider-hosted tool calls
  - Gemini GenerateContent response bridge now has explicit contract coverage for:
    - preserved aggregate usage totals plus `thoughtsTokenCount` / `cachedContentTokenCount`
    - preserved native `responseId` / `modelVersion`
    - preserved `groundingMetadata` / `urlContextMetadata` with source-grounding replay
    - documented lossy cases for generic STOP finish-reason collapse on client tool calls, prompt
      / completion audio breakdown, prediction-token breakdown, and dropped
      `system_fingerprint` / `service_tier`
    - same-protocol response encoding now reuses the shared Gemini content converter so bridge
      output preserves:
      - multi-part visible text segments
      - reasoning blocks
      - part-level `thoughtSignature`
      - provider-executed `code_execution` call/result pairs
    - initial same-protocol roundtrip fixture coverage now validates:
      - exact replay of visible text + reasoning partitioning with `thoughtSignature`
      - exact replay of provider-executed `code_execution` call/result pairs
      - documented lossy report behavior for client tool-call finish-reason collapse while still
        preserving tool-call semantics

## 5) Make streaming bridges explicit

- [x] Add explicit stream bridge adapters built on V3 stream parts
- [x] Ensure terminal events are preserved across protocol views
- [x] Ensure finish reasons survive target serialization
- [x] Ensure content block ordering is validated for Anthropic output
- [x] Ensure OpenAI final finish chunk behavior is consistent
- [x] Add no-network finalization tests for incomplete upstream termination
- [x] Ensure `BridgeMode::Strict` is enforced consistently for lossy stream routes
- [x] Add initial same-protocol stream roundtrip fixture coverage using semantic summaries instead
  of raw event-list equality
  - OpenAI Responses stream roundtrip coverage now validates:
    - response metadata
    - finish reason
    - prompt/completion totals
    - text reconstruction
    - reasoning boundary identity
  - Anthropic Messages stream roundtrip coverage now validates:
    - start metadata
    - finish reason
    - prompt/completion totals
    - text reconstruction
  - raw SSE event-for-event equality is intentionally not the target because same-protocol bridge
    serialization can legitimately regenerate protocol frames, timestamps, and duplicate deltas

## 6) Add customization hooks

- [x] Add bridge-scoped typed context types:
  - request
  - response
  - stream
  - primitive remap
- [x] Add object-safe hook contracts:
  - request pre-bridge transform
  - request post-bridge validation
  - response transform before target serialization
  - stream-part transform
- [x] Add an ergonomic bundled customization trait for multi-phase policies
  - `BridgeCustomization`
  - `BridgeOptions::with_customization(...)`
  - `BridgeOptionsOverride::with_customization(...)`
- [x] Add primitive remapper policy:
  - tool name remap
  - tool call id remap
  - tool choice remap
- [x] Add route-level override for `BridgeMode`
- [x] Add lossy field handling policy:
  - reject
  - warn and continue
  - provider-tolerant continue
- [x] Integrate customization with direct request pair bridges and normalized request bridges
- [x] Integrate customization with response and stream bridge paths
- [x] Provide closure-friendly wrappers in `siumai-extras`
- [x] Integrate protocol-source -> normalized request normalization with the same typed request
  customization surface
  - exposed through `*_json_to_chat_request_with_options(...)`
  - request hooks can now branch on `RequestBridgePhase::NormalizeSource` vs
    `RequestBridgePhase::SerializeTarget`
  - keep parser ownership protocol-owned; do not add a whole-parser override trait
- [x] Do not expose raw provider JSON patching as the default bridge extension story
  - documented as an escape hatch in the migration note and customization-boundary note
- [x] Add declarative provider-defined tool rewrite customization for direct-pair hosted-tool
  compatibility
  - `ProviderToolRewriteCustomization`
  - `ProviderToolRewriteRule`
  - argument remapping stays semantic and typed instead of patching final JSON

## 7) Add gateway runtime policy

- [x] Define `GatewayBridgePolicy` in `siumai-extras`
- [x] Cover:
  - [x] body limits
  - [x] upstream read limits
  - [x] stream idle timeout
  - [x] keepalive interval
  - [x] header filtering
  - [x] error passthrough
  - [x] bridge strictness default
  - [x] warning emission
- [x] Integrate gateway policy with bridge customization contracts instead of creating parallel
  hook types
- [x] Keep framework-agnostic pieces separate from Axum wrappers
- [x] Align Axum SSE transcode helpers with core stream loss-policy inspection when callers need
  strict inspected rejection on cross-protocol stream routes
  - enabled by explicit `TranscodeSseOptions::with_bridge_source(...)`
- [x] Let Axum JSON/SSE transcode helpers consume bridge customization options
- [x] Let Axum JSON/SSE transcode helpers consume `GatewayBridgePolicy`
- [x] Add symmetric gateway request-normalization helpers for provider-native request ingress

## 8) Documentation and examples

- [x] Add a bridge-focused architecture note after the contract lands
- [x] Document the hybrid bridge strategy:
  - normalized backbone
  - selected direct bridges
  - no full N x M pairwise mesh
- [x] Add a dedicated customization-boundary note covering:
  - typed hook / policy-based user-defined conversion
  - inbound normalization customization through explicit `with_options` bridge entry points
  - no whole-parser override trait
- [x] Document the recommended customization direction:
  - bridge-specific trait/policy objects in core
  - closure-friendly wrappers in `siumai-extras`
  - no raw JSON patch as the primary API
- [x] Document route-local closure-based customization for gateway helpers
- [x] Add runnable examples for:
  - [x] Anthropic -> OpenAI Responses gateway
  - [x] OpenAI Responses -> Anthropic gateway
  - [x] custom lossy-policy handling
  - [x] custom tool remapper
  - [x] custom stream transform
  - [x] pure bridge request customization bundle
- [x] Document the hosted-tool rewrite path explicitly:
  - direct pair curated hosted-tool translations remain in-tree
  - user-defined hosted-tool translation uses `ProviderToolRewriteCustomization`
  - gateway helpers reuse the same customization object instead of introducing gateway-only hooks
- [x] Extend the bridge customization example to show provider-hosted tool rewrite before direct
  pair translation
- [x] Update `docs/README.md` to include this workstream
- [x] Add a migration note if any public gateway helpers change shape

## 9) Validation

- [ ] Add fixture-based bridge tests for request, response, and streaming paths
  - OpenAI Responses request normalization now has initial fixture coverage for:
    - exact assistant/tool/provider-tool input cases
    - exact `text.format` structured-output restoration
    - best-effort `item_reference` restoration cases
    - known lossy `function_call_output`-only cases where tool name is not recoverable
  - OpenAI Chat Completions request normalization now has fixture coverage for:
    - exact system / assistant tool-call / tool-result / file input restoration cases
    - best-effort system-message-mode cases
    - response-format JSON schema restoration
  - OpenAI Chat Completions request bridge same-protocol roundtrip now has initial exact fixture
    coverage for:
    - system-message replay
    - assistant tool-call replay
    - assistant tool-result replay
    - structured-output response-format replay
    - documented rejected PDF file-id case for unsupported chat file media replay
  - Gemini GenerateContent request normalization now has initial coverage for:
    - system instruction / message / tool reconstruction
    - toolConfig function-calling and retrieval config restoration
    - same-protocol projected replay of `executableCode` / `codeExecutionResult`
    - structured-output provider option restoration via `responseJsonSchema`
  - Gemini GenerateContent request bridge same-protocol roundtrip now has initial exact fixture
    coverage for:
    - function tool replay
    - function tool-choice replay
    - empty tools omission replay
    - documented projected `googleSearch` replay with model uplift
  - Gemini GenerateContent response bridge same-protocol roundtrip now has fixture coverage for:
    - exact visible text + reasoning + `thoughtSignature` replay
    - exact provider-executed `code_execution` call/result replay
    - documented lossy client tool-call `finish_reason` collapse into generic `STOP` while keeping
      projected tool-call semantics
  - the same Gemini bridge surface is now available from `google-vertex` builds as well:
    - request normalization / request bridge / response bridge / stream bridge helpers no longer
      require enabling the separate `google` feature just to target Gemini wire format
    - Vertex chat response same-protocol roundtrip fixtures now cover the same projected/exact
      cases for reasoning, `thoughtSignature`, tool calls, and provider-executed
      `code_execution`
  - Anthropic Messages request normalization now has fixture coverage for:
    - base request settings
    - function tool choice and provider-defined tool restoration
    - structured output and MCP server options
    - thinking body/provider option restoration
  - Anthropic Messages -> OpenAI Responses request direct-pair fixture coverage now validates:
    - `web_search` and `code_execution` tool projection
    - Anthropic effort -> OpenAI reasoning effort projection
    - structured output `text.format` projection
  - OpenAI Responses -> Anthropic Messages request direct-pair fixture coverage now validates:
    - encrypted reasoning -> redacted thinking replay
    - system-message replay
    - explicit reporting for unsupported OpenAI local-shell tools
  - OpenAI Responses request bridge same-protocol roundtrip now has initial exact fixture coverage
    for:
    - visible text / assistant tool-call input replay
    - system-message mode projection
    - structured-output `text.format` replay
  - Anthropic Messages request bridge same-protocol roundtrip now has initial exact fixture
    coverage for:
    - base settings / tool-choice / provider-hosted tool replay
    - native structured-output `output_format` replay
    - thinking / MCP / context-management / effort overlays from provider options
    - container skill replay for code-execution skill environments
  - OpenAI Responses response roundtrip now has initial fixture coverage for:
    - exact same-protocol cases, including web-search source-citation replay and file-search
      source extraction from provider results
    - exact tool-scoped source reconstruction even when OpenAI provider item ids differ from
      unified tool-call ids
    - documented lossy message-citation source replay via annotations
  - Anthropic Messages response roundtrip now has initial fixture coverage for:
    - projected JSON output/tool cases
    - exact projected tool-search replay
  - OpenAI Responses stream roundtrip now has initial semantic-summary fixture coverage for:
    - text delta replay
    - reasoning start/end replay
  - Anthropic Messages stream roundtrip now has initial semantic-summary fixture coverage for:
    - JSON output text replay
    - reserved json-tool replay
- [x] Add explicit tests for lossy conversions
  - request:
    - default best-effort continues on lossy reasoning -> OpenAI Chat Completions conversion
  - response:
    - default strict rejects lossy usage-detail downgrade into Anthropic Messages
  - stream:
    - default best-effort continues on lossy cross-protocol stream routes while keeping
      `report.lossy_fields`
- [x] Add tests for custom hooks and primitive remappers
  - request hook coverage now exercises:
    - normalized request mutation
    - target JSON post-transform
    - target JSON validation/reporting
    - bundled `BridgeCustomization` request mutation + JSON overlay + validation + tool remap
  - response hook coverage now exercises:
    - response-side semantic rewrite before target serialization
    - bundled `BridgeCustomization` response rewrite + tool/call remap
  - stream customization coverage now exercises:
    - bundled `BridgeCustomization` stream event transform + tool/call remap
    - primitive remapping on `ToolCallDelta`
    - remapping propagation into `StreamEnd` response content
- [x] Add tests for strict vs best-effort behavior
  - request:
    - strict rejects lossy reasoning downgrade
    - best-effort continues on the same lossy route
  - response:
    - strict rejects lossy usage-detail downgrade by default
    - custom loss policy can still allow that route
  - stream:
    - strict rejects inspected cross-protocol lossy routes
    - best-effort continues on the same route
- [x] Add tests showing custom loss-policy behavior on request / response / stream bridges
- [x] Add gateway smoke coverage for JSON and SSE output paths
  - `siumai-extras` Axum integration now has router-level smoke coverage for:
    - OpenAI Responses JSON gateway output path
    - OpenAI Responses SSE gateway output path
    - Anthropic Messages JSON gateway output path
    - Anthropic Messages SSE gateway output path
    - fixture-backed Anthropic -> OpenAI Responses SSE route smoke
    - fixture-backed OpenAI Responses -> Anthropic Messages SSE route smoke
