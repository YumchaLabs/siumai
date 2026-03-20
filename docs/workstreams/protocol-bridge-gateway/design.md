# Protocol Bridge + Gateway Runtime - Design

Last updated: 2026-03-20

## Context

This workstream exists because `siumai` already has most of the right low-level ingredients for
cross-protocol gateway work, but those ingredients are still exposed through several separate
layers instead of one explicit bridge surface.

Today the workspace already has:

- provider-agnostic request/response/stream types
- typed V3 stream parts in the core runtime
- protocol-owned serializers and parsers in protocol crates
- gateway-oriented SSE/JSON transcoders in `siumai-extras`
- OpenAI WebSocket session support for incremental `/responses`
- provider-specific extensions for hosted tools, reasoning, prompt caching, and structured output

That direction is correct.

The bridge work since this document was first added has confirmed two additional points:

- bridge contracts should exist independently from protocol implementations
- explicit request bridge APIs are required even before pairwise direct bridges land

The current codebase now already has:

- typed bridge contracts in `siumai-core::bridge`
- an experimental explicit request bridge facade in `siumai::experimental::bridge`
- initial bridge reporting for lossy fields, dropped fields, and unsupported request semantics

The recent Anthropic/OpenAI streaming fixes confirmed the main architectural point: correctness now
depends on explicit end-of-stream semantics, event typing, loss handling, and protocol-owned state
machines, not only on "generic chat" request forwarding.

This workstream is inspired by two external reference lines:

- the Vercel AI SDK style of maintaining a normalized internal stream model and protocol-specific
  views
- `repo-ref/sub2api`, which makes some cross-protocol bridges explicit instead of hiding them
  inside route handlers

We will follow the architectural idea, not copy either implementation literally.

## Assessment of the current implementation

## What is already correct

- The internal normalized model is the right anchor.
  - `ChatRequest`, `ChatResponse`, `ChatStream`, and V3 stream parts are the correct place to
    preserve semantics before a protocol view is chosen.
- Protocol-specific serialization should stay protocol-owned.
  - Anthropic/OpenAI/Gemini wire contracts should continue to live in their protocol crates.
- `siumai-extras` is the right place for runnable gateway helpers.
  - Axum SSE/JSON transcoding and tool-loop gateway helpers already prove this boundary.
- Provider-specific capabilities should remain opt-in and typed where possible.
  - Prompt caching, reasoning, hosted tools, and provider-owned extensions are already moving in
    the right direction.

## What is missing

- There is no explicit "protocol bridge" package or module with a stable contract.
  - The logic is currently spread across serializers, converter state, and gateway helpers.
- Request bridging is not modeled as a first-class API.
  - We can serialize unified responses into other protocols more easily than we can bridge incoming
    request shapes across protocols in a reusable way.
- Lossy conversion is not yet a named contract.
  - Some conversions are exact, some are lossy, and some should be rejected, but that policy is
    not consistently represented in types.
- Customization is too narrow.
  - We already support some response transform hooks, but we do not yet have a unified request,
    response, and stream bridge hook story.
- Gateway runtime policy is not yet a stable layer.
  - Timeouts, header allowlists, strict vs best-effort behavior, error passthrough, stream
    keepalive, and loss reporting should be modeled intentionally instead of living only in route
    code.

## Goals

1. Introduce an explicit bidirectional protocol bridge surface.
2. Keep the normalized internal model as the semantic source of truth.
3. Make lossiness explicit, inspectable, and configurable.
4. Support request, response, and stream customization as first-class hooks.
5. Add a gateway runtime policy layer suitable for embeddable single-tenant or app-level gateways.

## Non-goals

- Do not turn `siumai` into a full Sub2API-style control plane.
- Do not add multi-tenant billing, subscriptions, admin dashboards, or account-pool scheduling to
  this workstream.
- Do not pretend all protocol conversions are exact.
- Do not force all public APIs to become protocol-specific; the normalized family APIs remain the
  recommended application surface.

## Design decisions

### D1 - Keep the normalized model as the architectural anchor

The bridge layer must not bypass the normalized runtime.

The conversion flow should be:

1. protocol request view -> normalized request or normalized stream parts
2. normalized runtime and provider execution
3. normalized response or stream parts -> protocol response view

This keeps provider semantics concentrated in one place and avoids route-specific ad hoc rewrites.

### D2 - Separate bridge primitives from gateway adapters

The work should be split across layers:

- protocol crates own protocol wire models, parsers, serializers, and event state machines
- `siumai-core` owns bridge contracts, loss reporting types, and protocol-view conversion traits
- `siumai-extras` owns gateway-ready adapters, route helpers, runtime policy types, and framework
  integrations

This keeps the core reusable without forcing Axum or route concerns into the protocol/runtime layer.

### D3 - Make bidirectional bridges explicit

We should expose bridge APIs intentionally instead of treating them as implementation detail.

Initial bridge targets should prioritize the protocols we already use most:

- Anthropic Messages <-> OpenAI Responses
- Anthropic Messages <-> OpenAI Chat Completions
- normalized response/stream -> Anthropic Messages / OpenAI Responses / OpenAI Chat Completions /
  Gemini GenerateContent

The first iteration does not need to cover every pairwise protocol combination directly.
Some conversions can remain "via normalized request/response" rather than "direct protocol A ->
protocol B" helpers.

### D3a - Do not build a full N x M direct-bridge matrix

We should not model protocol bridging as "every protocol gets a handwritten direct bridge to every
other protocol".

That approach creates too much glue code and pushes the maintenance burden into pairwise field
rewrites.

The recommended strategy is a hybrid model:

- the normalized request/response/stream model remains the main semantic backbone
- most routes bridge via the normalized model
- only a small number of high-value and high-loss protocol pairs get direct bridges

This is effectively a star topology with selected direct bridges, not a full mesh.

The first direct bridge pair should be:

- Anthropic Messages <-> OpenAI Responses

because both sides carry richer semantics than simpler chat-only formats:

- reasoning / thinking
- provider-hosted tools
- approval-oriented tool flows
- more stateful streaming behavior

If we route these through a thinner intermediate protocol view such as Chat Completions, we lose
too much fidelity.

### D4 - Represent loss and unsupported semantics in types

Bridges must produce a report, not only a payload.

Proposed model:

- `BridgeMode`
  - `Strict`
  - `BestEffort`
  - `ProviderTolerant`
- `BridgeReport`
  - warnings
  - lossy fields
  - dropped fields
  - unsupported capabilities
  - provider metadata carried through
- `BridgeDecision`
  - exact
  - lossy
  - rejected

Examples:

- Anthropic thinking -> OpenAI Chat Completions may be lossy
- OpenAI hosted tool result details -> Anthropic tool result may be lossy
- protocol-specific admin fields may be dropped but reported

### D5 - Support customization as a first-class contract

This workstream should explicitly support custom bridge behavior.

Required customization points:

- request pre-bridge transform
- request post-bridge validation
- response post-bridge transform
- stream-part transform before target serialization
- tool name / tool id remapping
- strictness override per route or per request
- custom handling of lossy fields

This is important because "one canonical bridge" will not fit all gateways.
Some gateways want exact protocol mimicry.
Some want best-effort compatibility.
Some want provider-specific enrichment or filtering.

### D5a - Prefer reusable bridge primitives over pairwise glue code

Direct bridges are still necessary for a few protocol pairs, but they must not become large
pairwise monoliths.

The implementation should be split into reusable bridge primitives:

- reasoning policy
- tool definition mapping
- tool call mapping
- tool result mapping
- cache control handling
- approval handling
- provider metadata carry/drop policy
- stream finalization policy

Pairwise bridges should compose these primitives instead of owning all field-level logic
themselves.

That keeps direct bridges maintainable and prevents every new protocol pair from becoming a new
copy-paste conversion layer.

### D6 - Introduce a gateway runtime policy layer

`siumai-extras` should expose a stable gateway policy object rather than only narrow helper
functions.

The policy layer should cover:

- request body limits
- upstream read limits
- stream keepalive and idle timeout
- header allowlist / denylist
- error passthrough rules
- protocol strictness defaults
- bridge warning emission
- whether lossy conversions fail or continue
- optional custom per-route hooks

This layer is intentionally smaller than a full control plane.
It is meant for embeddable gateways inside Rust applications.

### D7 - Streaming correctness is part of the bridge contract

Stream bridges must be treated as stateful protocol machines, not as line-by-line text rewrites.

Requirements:

- terminal events must be explicit
- finish reasons must survive protocol translation
- content block start/delta/stop ordering must remain valid
- repeated or synthetic events must be deduplicated intentionally
- end-of-stream finalization must be testable without a network

The recent Anthropic SSE fixes fall directly under this rule.

### D8 - Separate planner, pair bridge, and gateway policy

The public bridge surface should have three distinct responsibilities:

- planner
  - decides whether a conversion should use a direct bridge, a normalized path, or rejection
- bridge implementation
  - request / response / stream bridge code that performs the actual conversion
- gateway policy
  - route/runtime concerns such as strictness defaults, warning emission, headers, and limits

These responsibilities must stay separate.

In particular:

- gateway code must not embed pairwise conversion logic
- pairwise bridges must not embed route-level policy
- the planner must not own protocol wire serialization

### D9 - Request, response, and stream bridges are different systems

The workstream must not collapse request, response, and stream bridges into one generic
"protocol bridge" implementation.

They share contracts and reports, but they have different correctness constraints:

- request bridges care about request shape, tool declarations, prompt caching, and structured input
- response bridges care about usage, finish semantics, and output shape
- stream bridges care about ordering, terminal events, deduplication, and finalization

They should therefore reuse contracts and primitives, but still live in separate implementation
modules.

## Proposed public shape

The exact module names can change, but the shape should look roughly like this:

```rust
pub enum BridgeTarget {
    OpenAiResponses,
    OpenAiChatCompletions,
    AnthropicMessages,
    GeminiGenerateContent,
}

pub struct BridgeOptions {
    pub mode: BridgeMode,
    pub report_warnings: bool,
    pub hooks: BridgeHooks,
}

pub struct BridgeHooks {
    pub request_transform: Option<...>,
    pub response_transform: Option<...>,
    pub stream_transform: Option<...>,
}

pub struct BridgeResult<T> {
    pub value: T,
    pub report: BridgeReport,
}
```

Possible surface split:

- low-level conversion:
  - normalized request/response/stream parts -> target protocol view
- protocol compat helpers:
  - `anthropic_to_openai_responses(...)`
  - `openai_responses_to_anthropic(...)`
- gateway helpers:
  - `to_transcoded_sse_response(...)`
  - `to_transcoded_json_response(...)`
  - future route-policy wrappers using `GatewayBridgePolicy`

Recommended implementation split:

- `siumai-core::bridge`
  - bridge contracts and reports only
- `siumai::experimental::bridge`
  - planner and facade entry points
- `siumai::experimental::bridge::request`
  - explicit request bridges
- `siumai::experimental::bridge::response`
  - explicit non-streaming response bridges
- `siumai::experimental::bridge::stream`
  - explicit stream bridges
- protocol crates
  - protocol-owned wire types, serializers, parsers, and state machines

Recommended internal shape for request bridges:

- `planner`
- `request::primitives`
- `request::pairs`
- `request::inspect`
- `request::serialize`

`pairs` should compose `primitives`; they should not become field-mapping dumping grounds.

## Recommended implementation order

1. Define bridge contracts and loss-reporting types.
2. Refactor request bridge code into planner + primitives + pair modules.
3. Stabilize explicit normalized request -> target protocol request bridges.
4. Add the first direct pair bridge: Anthropic Messages <-> OpenAI Responses.
5. Stabilize non-streaming response bridges.
6. Stabilize streaming bridges and finalization semantics.
7. Add gateway runtime policy objects and route helpers.
8. Expand customization hooks.

## Success criteria

This workstream is successful when:

- a user can explicitly bridge request, response, or stream shapes across protocols
- lossy behavior is reported and configurable
- gateway helpers are built on top of the bridge layer, not vice versa
- protocol semantics are preserved by protocol-owned state machines
- custom hooks are supported without patching internal converter code
