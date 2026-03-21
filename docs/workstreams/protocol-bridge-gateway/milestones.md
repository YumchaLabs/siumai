# Protocol Bridge + Gateway Runtime - Milestones

Last updated: 2026-03-21

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
- initial same-protocol response roundtrip fixture coverage now exists for OpenAI Responses and
  Anthropic Messages
- that coverage is intentionally split into exact, projected, and documented-lossy cases rather
  than claiming full wire-level losslessness
- remaining work is mainly cross-target usage / reasoning / structured-output audit

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
- protocol-source request normalization is now explicit, and its extension story is intentionally
  wrapper-first instead of exposing a whole-parser override trait
- `BridgeOptionsOverride` and gateway helper mode-override entry points now exist for route-level
  strictness/customization without rebuilding full bridge options
- `siumai-extras` Axum JSON/SSE transcode helpers now accept bridge customization, and closure-
  friendly adapters exist for request/response/stream hooks, loss-policy hooks, and primitive
  remappers
- request / response / stream tests now also cover:
  - custom loss-policy behavior beyond the default mode-aware policy
  - explicit lossy conversion paths across request / response / stream bridges
  - strict-vs-best-effort behavior on representative lossy routes
  - request hook mutation / JSON validation
  - response hook mutation
  - stream primitive remapping on delta and final response paths
- the customization boundary is now explicitly documented:
  - typed bridge hooks and policies are the supported user-defined conversion surface
  - wrapper-composed post-normalize transforms are preferred over a whole-parser override trait

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
- runnable bridge demos now exist for:
  - Anthropic Messages request normalization -> OpenAI Responses JSON/SSE output
  - OpenAI Responses request normalization -> Anthropic Messages JSON/SSE output
- `gateway-loss-policy` now demonstrates:
  - default strict rejection on lossy bridge routes
  - allowlisted `BridgeLossPolicy` continuation
  - full continue-policy override for comparison
  - the same `siumai-extras` JSON/SSE helper surface instead of mixing Axum helpers with direct
    core stream bridge wiring
- gateway migration guidance now exists at:
  - `docs/workstreams/protocol-bridge-gateway/migration.md`
- request normalization now has fixture-based coverage across the four currently supported source
  protocols:
  - OpenAI Responses exact and best-effort lossy restoration
  - OpenAI Chat Completions exact and best-effort request restoration
  - Gemini GenerateContent projected request restoration and same-protocol roundtrip coverage
  - Anthropic Messages settings / tools / structured output / MCP / thinking restoration
- response/stream fixture expansion has now started with initial same-protocol roundtrip coverage
  for OpenAI Responses and Anthropic Messages
- `siumai-extras` now also has Axum router-level smoke coverage for:
  - OpenAI Responses JSON/SSE output routes
  - Anthropic Messages JSON/SSE output routes
  - fixture-backed Anthropic -> OpenAI Responses SSE route transcoding
  - fixture-backed OpenAI Responses -> Anthropic Messages SSE route transcoding
- remaining work is mainly second-route examples, broader cross-target fixture expansion, and
  stabilization docs

Status: in progress
