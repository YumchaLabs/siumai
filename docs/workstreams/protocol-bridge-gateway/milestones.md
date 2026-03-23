# Protocol Bridge + Gateway Runtime - Milestones

Last updated: 2026-03-23

This workstream is tracked by milestones with explicit acceptance criteria.

## PBG-M0 - Scope and boundary locked

Acceptance criteria:

- The workstream scope is explicitly documented.
- The package boundary between protocol crates, `siumai-core`, and `siumai-extras` is fixed.
- Non-goals clearly exclude billing, subscriptions, admin dashboards, and account-pool control
  plane work.

Status: completed

## PBG-M1 - Bridge contracts exist

Acceptance criteria:

- `BridgeTarget`, `BridgeMode`, `BridgeReport`, and `BridgeResult<T>` exist.
- Lossy conversion and unsupported semantics are represented in types.
- No-network tests verify report behavior for exact, lossy, and rejected cases.

Status: completed

## PBG-M2 - Request bridges are explicit

Acceptance criteria:

- Explicit request bridge APIs exist for normalized request -> target protocol request conversion.
- At least the highest-value direct compat path exists:
  - Anthropic Messages <-> OpenAI Responses
- Request bridges emit `BridgeReport`.
- `Strict` mode rejects unsupported input semantics.

Current state:

- normalized request -> target protocol JSON APIs exist
- protocol-source typed request -> normalized request APIs now exist for Anthropic Messages,
  Gemini GenerateContent, OpenAI Responses, and OpenAI Chat Completions
- request bridge reports and strict rejection exist
- the direct Anthropic Messages <-> OpenAI Responses request bridge exists

Status: completed

## PBG-M2a - Request bridge structure is reusable

Acceptance criteria:

- request bridge code is split into planner, primitives, and pair bridge modules
- reusable reasoning / tools / cache / approval inspection logic is shared across targets
- adding a new direct pair bridge does not require editing one large request bridge file

Status: completed

## PBG-M3 - Non-streaming response bridges are explicit

Acceptance criteria:

- Normalized responses can be converted into target protocol response views through explicit APIs.
- Tool calls, usage, finish semantics, and structured output survive when representable.
- Lossy response conversion is reported consistently.

Current state:

- explicit normalized response -> target response JSON APIs exist
- loss reporting and no-network response bridge tests exist
- OpenAI Responses fidelity now preserves provider-executed tool items and response sources where
  representable
- OpenAI Responses same-protocol response roundtrip now also preserves normalized web-search
  embedded source payloads, file-search typed source extraction from raw provider results, and
  tool-scoped source id / linkage reconstruction when OpenAI item ids differ from unified
  tool-call ids
- Anthropic Messages same-protocol response roundtrip now also preserves provider-hosted
  tool-search tool-result reconstruction, raw server tool names, caller metadata, and MCP
  server-name metadata when those fields are carried on tool-call metadata
- Gemini GenerateContent fidelity now also preserves native response metadata that was previously
  being dropped by the non-streaming encoder:
  - `responseId` / `modelVersion`
  - `cachedContentTokenCount`
  - `groundingMetadata` / `urlContextMetadata`
  - source-grounding replay via preserved grounding metadata or normalized source parts
- Gemini GenerateContent same-protocol response encoding now also reuses the shared Gemini content
  conversion path, so fixture-backed roundtrip coverage can validate:
  - exact multi-part visible text + reasoning replay
  - exact part-level `thoughtSignature` replay
  - exact provider-executed `code_execution` call/result replay
  - documented lossy report behavior for client tool-call `finish_reason` collapse into generic
    `STOP`
- the same Gemini bridge helpers are now available under `google-vertex` builds, so Vertex users
  do not need the separate `google` feature just to access:
  - Gemini request normalization
  - Gemini request/response bridge helpers
  - Gemini stream bridge helpers
- Vertex same-protocol request roundtrip coverage now also documents that provider-hosted tools
  such as `google_search` and `code_execution` continue to follow the shared Gemini
  target-capability projection path, including model uplift, rather than a Vertex-only exact
  replay path
- Vertex same-protocol response roundtrip fixture coverage now mirrors the Gemini cases for:
  - reasoning + `thoughtSignature`
  - provider-executed `code_execution`
  - documented client tool-call `finish_reason` projection
- Gemini family same-protocol stream roundtrip fixture coverage now exists for both Google and
  Vertex decoders, including:
  - provider-executed `code_execution` call/result replay
  - reasoning `thoughtSignature` replay with provider namespace preservation
- Gemini stream serialization now keeps provider-hosted `code_execution` tool calls on the
  `executableCode` path and preserves reasoning `thoughtSignature` without emitting duplicate
  thinking chunks during same-protocol replay
- OpenAI Chat Completions same-protocol stream roundtrip fixture coverage now exists for:
  - assistant role/text/finish replay
  - tool-call argument accumulation plus usage replay
  - multiple tool-call replay
  - Azure model-router style chunks without synthetic start metadata injection
- target-level reasoning / structured-output / usage audit is now recorded in `audit.md`, including:
  - structured output being primarily a request-target concern on Anthropic / OpenAI / Gemini paths
  - OpenAI Responses as the strongest current target for modeled reasoning + usage detail fidelity
  - OpenAI Chat Completions as structurally lossy for reasoning and totals-only on the stream
    usage surface
  - Gemini GenerateContent as projected-exact for modeled same-protocol reasoning +
    `thoughtSignature` + provider-executed `code_execution`, while richer usage breakdown remains
    lossy
- initial same-protocol response roundtrip fixture coverage now exists for OpenAI Responses and
  Anthropic Messages, and Gemini GenerateContent
- that coverage is intentionally split into exact, projected, and documented-lossy cases rather
  than claiming full wire-level losslessness
- remaining work is now mainly broader fixture expansion and selective gap closure

Status: in progress

## PBG-M4 - Streaming bridges are explicit and stateful

Acceptance criteria:

- Streaming bridge adapters are built on V3 stream parts and protocol-owned state machines.
- Terminal events and finish reasons remain correct across target protocols.
- Anthropic block ordering and OpenAI final finish chunk behavior are covered by no-network tests.
- Finalization logic is explicit for incomplete upstream stream termination.

Current state:

- explicit stream bridge adapters already exist
- protocol-owned event converters are already on the serialization path
- target serializers now preserve finish reasons in terminal protocol frames across Anthropic,
  OpenAI-compatible, OpenAI Responses, and Gemini stream views
- Anthropic target serialization now closes and reopens content blocks on block-kind switches so
  interleaved text / thinking / tool segments remain protocol-valid, and bridge tests cover the
  same behavior end to end
- clean EOF finalization is now explicit in bridge and Axum SSE serialization paths, and no-network
  tests cover incomplete upstream termination
- OpenAI Chat Completions terminal serialization now uses one shared final-chunk path for both
  V3 `finish` parts and `StreamEnd`, and V3 `response-metadata` now seeds the terminal chunk state
  so bridged final frames keep the upstream response id / model / timestamp
- stream bridge inspection now records cross-protocol best-effort normalization as a lossy route,
  so `BridgeMode::Strict` rejects those conversions consistently while same-protocol strict routes
  still pass
- same-protocol OpenAI Responses and Anthropic Messages stream roundtrip fixtures now validate
  semantic summaries instead of raw event-for-event equality because target serializers may
  legitimately regenerate protocol frames, timestamps, and duplicate deltas during replay
- OpenAI Responses same-protocol stream roundtrip coverage has since been expanded to include:
  - apply-patch
  - code-interpreter
  - file-search
  - image-generation
  - local-shell
  - MCP tool + approval flows
  - shell
  - finish-carried logprobs
- the OpenAI Responses stream serializer now also preserves two previously exposed same-protocol
  replay gaps:
  - MCP provider tool-call replay
  - finish-carried output-text logprobs replay
- protocol-crate public feature-surface integration tests now also exist for:
  - Anthropic Messages SSE under `anthropic-standard`
  - OpenAI Responses SSE under `openai-standard` + `openai-responses`
  - Gemini streaming under `google`
  - these tests validate the exported converter surfaces directly instead of relying only on
    internal module tests

Status: completed

## PBG-M5 - Bridge customization contracts exist

Acceptance criteria:

- `BridgeOptions` exists as the stable configuration surface for bridge customization.
- Typed request / response / stream context types exist.
- Object-safe bridge hooks exist for request, response, and stream customization.
- Primitive remapper policy exists for tool names and tool call IDs.
- Lossy handling policy is customizable and tested.

Current state:

- `BridgeOptions`, typed contexts, object-safe hooks, primitive remapper, and loss-policy traits
  now exist in `siumai-core::bridge`
- experimental request / response / stream bridge entry points now consume bridge customization
  options
- protocol-source request normalization now also exposes `with_options` entry points so inbound
  normalization can reuse the same typed request hook/remapper/loss-policy surface without
  exposing a whole-parser override trait
  - `BridgeOptionsOverride` and gateway helper mode-override entry points now exist for route-level
    strictness/customization without rebuilding full bridge options
- `siumai-extras::bridge` now exposes framework-neutral closure-friendly adapters for
  request/response/stream hooks, loss-policy hooks, and primitive remappers
- `siumai-extras` Axum JSON/SSE transcode helpers now accept the same bridge customization surface
- request / response / stream tests now also cover:
  - custom loss-policy behavior beyond the default mode-aware policy
  - explicit lossy conversion paths across request / response / stream bridges
  - strict-vs-best-effort behavior on representative lossy routes
  - request hook mutation / JSON validation
  - bundled response customization plus tool/call remapping
  - bundled stream customization plus delta/final-response tool remapping
- the customization boundary is now explicitly documented:
  - typed bridge hooks and policies are the supported user-defined conversion surface
  - inbound normalization now uses the same hook surface through explicit `with_options` entry
    points instead of route-local parser forks
- `ProviderToolRewriteCustomization` and `ProviderToolRewriteRule` now provide a small,
  declarative customization layer for provider-hosted tool rewrites before direct pair
  translation
- direct pair hosted-tool rule tables and OpenAI MCP -> Anthropic `mcp_servers` helpers are now
  isolated from pair-specific glue so new compat work does not require editing one large file

Status: completed

## PBG-M6 - Gateway runtime policy exists

Acceptance criteria:

- `siumai-extras` exposes a stable gateway bridge policy layer.
- JSON and SSE gateway helpers can be configured with strictness, error passthrough, header policy,
  keepalive, timeout behavior, and bridge customization.
- The policy layer does not require embedding route-specific ad hoc conversion logic.

Dependency note:

- This milestone should build on PBG-M5 instead of inventing a second hook system.

Current state:

- `GatewayBridgePolicy` now exists in `siumai-extras::server`
- framework-agnostic bridge/policy helpers now live under `siumai-extras::server` instead of the
  Axum adapters:
  - bridge option resolution
  - bridge diagnostic header emission
  - SSE runtime timer policy projection
- Axum JSON/SSE transcode helpers can already consume `GatewayBridgePolicy`
- bridge decision / warning headers are now emitted for JSON routes and bridge target / mode
  headers are emitted for SSE routes when enabled
- Axum SSE transcode helpers now enforce keepalive interval and idle-timeout behavior directly from
  `GatewayBridgePolicy`
- Axum runtime helpers now enforce request-body and upstream-read limits for body / JSON reads
- Axum SSE transcode helpers now also run inspected stream loss-policy rejection and emit
  decision/warning headers when callers provide an explicit upstream protocol via
  `TranscodeSseOptions::with_bridge_source(...)`

Status: completed

## PBG-M7 - Examples and stabilization

Acceptance criteria:

- Runnable examples exist for at least two bridge routes and one custom hook or remapper route.
- `docs/README.md` and related gateway docs point to the workstream.
- Bridge behavior is covered by fixture-based tests and gateway smoke tests.
- The resulting public story is clear:
  - normalized runtime first
  - explicit bridge surface second
  - gateway adapter layer third

Current state:

- `gateway-custom-transform` now demonstrates `GatewayBridgePolicy`, typed response/stream hooks,
  and primitive remappers as the primary customization path
- `bridge-customization` now demonstrates the reusable pure bridge path with:
  - `BridgeCustomization`
  - `BridgeOptions::with_customization(...)`
  - request mutation, target JSON overlay, JSON validation, and tool-name remapping in one object
- `bridge-customization` now also demonstrates provider-hosted tool rewrite using
  `ProviderToolRewriteCustomization` so cross-protocol hosted-tool conversion does not require
  request JSON patch glue
- runnable bridge demos now exist for:
  - Anthropic Messages request normalization -> OpenAI Responses JSON/SSE output
  - OpenAI Responses request normalization -> Anthropic Messages JSON/SSE output
- the Anthropic Messages -> OpenAI Responses gateway example now also demonstrates request-side
  hosted-tool rewrite through `GatewayBridgePolicy::with_customization(...)`, so the same
  customization object flows through gateway request normalization and downstream bridge helpers
- `gateway-loss-policy` now demonstrates:
  - default strict rejection on lossy bridge routes
  - allowlisted `BridgeLossPolicy` continuation
  - full continue-policy override for comparison
  - the same `siumai-extras` JSON/SSE helper surface instead of mixing Axum helpers with direct
    core stream bridge wiring
- gateway migration guidance now exists at:
  - `docs/workstreams/protocol-bridge-gateway/migration.md`
- the customization note now documents:
  - curated in-tree hosted-tool translations in direct pair bridges
  - user-defined hosted-tool rewrite via `ProviderToolRewriteCustomization`
  - gateway reuse of the same customization objects through policy/route helper entry points
- request normalization now has fixture-based coverage across the four currently supported source
  protocols:
  - OpenAI Responses exact and best-effort lossy restoration
  - OpenAI Chat Completions exact and best-effort request restoration
  - OpenAI Chat Completions same-protocol request bridge roundtrip coverage for system-message,
    assistant tool-call/tool-result, structured-output response-format, and documented rejected
    PDF file-id replay
  - OpenAI Chat Completions same-protocol response bridge fixture roundtrip coverage for legacy
    `function_call`, assistant text + tool-call replay, usage details, and preserved
    `system_fingerprint` / `service_tier`
  - Gemini GenerateContent projected request restoration and initial same-protocol request bridge
    fixture roundtrip coverage for function tools, function tool-choice, and empty-tools omission,
    plus documented projected `googleSearch` replay with model uplift
  - Gemini GenerateContent same-protocol response bridge fixture roundtrip coverage for
    multi-part visible text + reasoning replay, part-level `thoughtSignature`, provider-executed
    `code_execution` call/result replay, and documented client tool-call `finish_reason`
    projection
  - the same Gemini bridge helper surface is now exposed from `google-vertex` builds, with Vertex
    response roundtrip fixture coverage for the same reasoning/tool/code-execution cases
  - Anthropic Messages settings / tools / structured output / MCP / thinking restoration
- response/stream fixture expansion has now started with initial same-protocol roundtrip coverage
  for OpenAI Responses, Anthropic Messages, and Gemini GenerateContent
- `siumai-extras` now also has Axum router-level smoke coverage for:
  - OpenAI Responses JSON/SSE output routes
  - Anthropic Messages JSON/SSE output routes
  - Gemini GenerateContent JSON/SSE output routes
  - fixture-backed Anthropic -> OpenAI Responses SSE route transcoding
  - fixture-backed OpenAI Responses -> Anthropic Messages SSE route transcoding
  - fixture-backed Gemini -> OpenAI Responses SSE route transcoding
  - fixture-backed OpenAI Responses -> Gemini GenerateContent SSE route transcoding
  - Gemini target strict / best-effort bridge-decision header behavior on cross-protocol SSE
- remaining work is mainly second-route examples, broader cross-target fixture expansion, and
  stabilization docs

Status: in progress
