# Protocol Bridge + Gateway Runtime - Milestones

Last updated: 2026-03-20

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
- remaining work is strict-mode enforcement, finish-reason fidelity, block ordering validation, and
  incomplete upstream finalization tests

Status: in progress

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
- `siumai-extras` Axum JSON/SSE transcode helpers now accept bridge customization, and closure-
  friendly adapters exist for response hooks, stream hooks, and primitive remappers
- remaining work is mainly custom loss-policy coverage, gateway-policy composition, and broader
  examples

Status: in progress

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
- Axum JSON/SSE transcode helpers can already consume `GatewayBridgePolicy`
- bridge decision / warning headers are now emitted for JSON routes and bridge target / mode
  headers are emitted for SSE routes when enabled
- Axum SSE transcode helpers now enforce keepalive interval and idle-timeout behavior directly from
  `GatewayBridgePolicy`
- remaining work is mainly request-body / upstream-read limit ownership at route-runtime level and
  broader framework-agnostic integration beyond the current helper layer

Status: in progress

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
- remaining work is mainly second-route examples, fixture-based bridge coverage, and stabilization
  docs

Status: in progress
