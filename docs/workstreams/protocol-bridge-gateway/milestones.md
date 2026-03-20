# Protocol Bridge + Gateway Runtime - Milestones

Last updated: 2026-03-20

This workstream is tracked by milestones with explicit acceptance criteria.

## PBG-M0 - Scope and boundary locked

Acceptance criteria:

- The workstream scope is explicitly documented.
- The package boundary between protocol crates, `siumai-core`, and `siumai-extras` is fixed.
- Non-goals clearly exclude billing, subscriptions, admin dashboards, and account-pool control
  plane work.

Status: planned

## PBG-M1 - Bridge contracts exist

Acceptance criteria:

- `BridgeTarget`, `BridgeMode`, `BridgeReport`, and `BridgeResult<T>` exist.
- Lossy conversion and unsupported semantics are represented in types.
- No-network tests verify report behavior for exact, lossy, and rejected cases.

Status: planned

## PBG-M2 - Request bridges are explicit

Acceptance criteria:

- Anthropic Messages, OpenAI Responses, and OpenAI Chat Completions inbound shapes can be bridged
  through explicit APIs.
- At least the highest-value direct compat path exists:
  - Anthropic Messages <-> OpenAI Responses
- Request bridges emit `BridgeReport`.
- `Strict` mode rejects unsupported input semantics.

Status: planned

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

