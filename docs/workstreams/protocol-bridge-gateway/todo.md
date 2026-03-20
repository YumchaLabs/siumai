# Protocol Bridge + Gateway Runtime - TODO

Last updated: 2026-03-20

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

- [ ] Document the current inbound and outbound paths for:
  - Anthropic Messages
  - OpenAI Responses
  - OpenAI Chat Completions
  - Gemini GenerateContent
- [ ] Mark which conversions are already exact, lossy, or implicit
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
  - request / response / stream context types
  - primitive remapper context
  - lossy handling policy trait

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
- [ ] Add explicit protocol-source typed request -> normalized request bridge entry points for:
  - Anthropic Messages -> normalized request
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

## 4) Make non-streaming response bridges explicit

- [x] Add explicit normalized response -> target protocol converters
- [x] Emit loss reports for dropped or downgraded semantics
- [x] Add no-network tests for exact and lossy cases
- [x] Preserve tool calls, provider-executed tool results, and OpenAI response sources where
  representable
- [ ] Audit remaining reasoning / structured output / usage fidelity gaps per target

## 5) Make streaming bridges explicit

- [x] Add explicit stream bridge adapters built on V3 stream parts
- [x] Ensure terminal events are preserved across protocol views
- [ ] Ensure finish reasons survive target serialization
- [ ] Ensure content block ordering is validated for Anthropic output
- [ ] Ensure OpenAI final finish chunk behavior is consistent
- [ ] Add no-network finalization tests for incomplete upstream termination
- [ ] Ensure `BridgeMode::Strict` is enforced consistently for lossy stream routes

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
- [x] Add primitive remapper policy:
  - tool name remap
  - tool call id remap
  - tool choice remap
- [ ] Add route-level override for `BridgeMode`
- [ ] Add lossy field handling policy:
  - reject
  - warn and continue
  - provider-tolerant continue
- [x] Integrate customization with direct request pair bridges and normalized request bridges
- [x] Integrate customization with response and stream bridge paths
- [x] Provide closure-friendly wrappers in `siumai-extras`
- [ ] Do not expose raw provider JSON patching as the default bridge extension story

## 7) Add gateway runtime policy

- [x] Define `GatewayBridgePolicy` in `siumai-extras`
- [ ] Cover:
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
- [ ] Keep framework-agnostic pieces separate from Axum wrappers
- [x] Let Axum JSON/SSE transcode helpers consume bridge customization options
- [x] Let Axum JSON/SSE transcode helpers consume `GatewayBridgePolicy`

## 8) Documentation and examples

- [x] Add a bridge-focused architecture note after the contract lands
- [x] Document the hybrid bridge strategy:
  - normalized backbone
  - selected direct bridges
  - no full N x M pairwise mesh
- [x] Document the recommended customization direction:
  - bridge-specific trait/policy objects in core
  - closure-friendly wrappers in `siumai-extras`
  - no raw JSON patch as the primary API
- [ ] Add runnable examples for:
  - Anthropic -> OpenAI Responses gateway
  - OpenAI Responses -> Anthropic gateway
  - custom lossy-policy handling
  - [x] custom tool remapper
  - [x] custom stream transform
- [x] Update `docs/README.md` to include this workstream
- [ ] Add a migration note if any public gateway helpers change shape

## 9) Validation

- [ ] Add fixture-based bridge tests for request, response, and streaming paths
- [ ] Add explicit tests for lossy conversions
- [ ] Add tests for custom hooks and primitive remappers
- [ ] Add tests for strict vs best-effort behavior
- [ ] Add gateway smoke coverage for JSON and SSE output paths
