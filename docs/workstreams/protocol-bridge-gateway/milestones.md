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
- the direct Anthropic Messages <-> OpenAI Responses request bridge is not implemented yet
- the implementation is being refactored toward planner + primitives + pair modules

Status: in progress

## PBG-M2a - Request bridge structure is reusable

Acceptance criteria:

- request bridge code is split into planner, primitives, and pair bridge modules
- reusable reasoning / tools / cache / approval inspection logic is shared across targets
- adding a new direct pair bridge does not require editing one large request bridge file

Status: in progress

## PBG-M3 - Non-streaming response bridges are explicit

Acceptance criteria:

- Normalized responses can be converted into target protocol response views through explicit APIs.
- Tool calls, usage, finish semantics, and structured output survive when representable.
- Lossy response conversion is reported consistently.

Status: planned

## PBG-M4 - Streaming bridges are explicit and stateful

Acceptance criteria:

- Streaming bridge adapters are built on V3 stream parts and protocol-owned state machines.
- Terminal events and finish reasons remain correct across target protocols.
- Anthropic block ordering and OpenAI final finish chunk behavior are covered by no-network tests.
- Finalization logic is explicit for incomplete upstream stream termination.

Status: planned

## PBG-M5 - Gateway runtime policy exists

Acceptance criteria:

- `siumai-extras` exposes a stable gateway bridge policy layer.
- JSON and SSE gateway helpers can be configured with strictness, error passthrough, header policy,
  keepalive, and timeout behavior.
- The policy layer does not require embedding route-specific ad hoc logic.

Status: planned

## PBG-M6 - Customization is first-class

Acceptance criteria:

- Request, response, and stream bridge hooks are supported.
- Tool name / id remapping is supported.
- Route-level strict vs best-effort override is supported.
- Lossy field handling policy is configurable and tested.

Status: planned

## PBG-M7 - Examples and stabilization

Acceptance criteria:

- Runnable examples exist for at least two bridge routes and one custom hook route.
- `docs/README.md` and related gateway docs point to the workstream.
- Bridge behavior is covered by fixture-based tests and gateway smoke tests.
- The resulting public story is clear: normalized runtime first, explicit bridge surface second,
  gateway adapter layer third.

Status: planned
