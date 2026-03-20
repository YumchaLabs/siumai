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

The bridge work since this document was first added has confirmed three additional points:

- bridge contracts should exist independently from protocol implementations
- explicit request/response/stream bridge APIs are required instead of route-local glue
- bridge customization must be bridge-aware, not only "mutate some JSON before send"

The current codebase now already has:

- typed bridge contracts in `siumai-core::bridge`
- an experimental explicit bridge facade in `siumai::experimental::bridge`
- a request bridge planner plus direct pair bridge modules
- initial bridge reporting for lossy fields, dropped fields, and unsupported request semantics
- bridge-owned customization through `BridgeOptions`, typed contexts, hook traits, primitive
  remappers, and loss-policy traits
- gateway helper transforms and policy composition in `siumai-extras` for JSON and SSE routes

The recent Anthropic/OpenAI request and response roundtrip fixes confirmed the main architectural
point: correctness now depends on explicit end-of-stream semantics, event typing, loss handling,
provider-executed tool preservation, source preservation, and protocol-owned state machines, not
only on "generic chat" request forwarding.

The most recent streaming work also confirmed one more rule: clean EOF finalization must be an
explicit bridge/runtime step. A bridge or gateway serializer cannot assume every upstream source
will emit a protocol-level terminal event. We now handle that explicitly before target SSE
encoding instead of scattering protocol-specific EOF glue across serializers.

This workstream is inspired by two external reference lines:

- the Vercel AI SDK style of maintaining a normalized internal model and protocol-specific views
- `repo-ref/sub2api`, which makes some cross-protocol bridges explicit instead of hiding them
  inside route handlers

We will follow the architectural idea, not copy either implementation literally.

## Assessment of the current implementation

### What is already correct

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
- The hybrid bridge strategy is already proving itself.
  - The normalized model remains the backbone while a small number of direct pair bridges reduce
    loss where that cost is justified.

### What is missing

- Bridge customization is now available, but it still needs broader examples and stricter guidance.
  - `BridgeOptions` and typed hook traits now exist, but the public story still needs more runnable
    examples and clearer migration guidance away from ad hoc JSON patching.
- Lossy conversion policy now has a stable contract, but coverage is still incomplete.
  - The `BridgeLossPolicy` trait exists, but most production behavior is still exercised through
    default mode-driven policies and needs more custom-policy validation.
- Legacy customization points still sit adjacent to the bridge.
  - Existing execution and gateway hooks remain useful escape hatches, but bridge-owned typed hooks
    should become the clearly documented primary extension path.
- Gateway runtime policy now exists for Axum, but framework-agnostic factoring is still incomplete.
  - Request body limits, upstream read limits, keepalive, idle timeout, and header policy now have
    one runtime home, but broader non-Axum ownership is still future work.

### Current customization surface today

The repository now exposes both bridge-owned customization and older adjacent hooks.

Recommended primary surface:

- `siumai-core::bridge::{BridgeOptions, RequestBridgeHook, ResponseBridgeHook, StreamBridgeHook}`
- `siumai-core::bridge::{BridgePrimitiveRemapper, BridgeLossPolicy}`
- typed contexts:
  - `RequestBridgeContext`
  - `ResponseBridgeContext`
  - `StreamBridgeContext`
  - `BridgePrimitiveContext`
- `siumai-extras::server::GatewayBridgePolicy`
- closure-friendly adapters in `siumai-extras` for route-local response/stream transforms and
  primitive remapping

Legacy but still useful adjacent hooks:

- Request construction and provider body hooks already exist:
  - `siumai-core::execution::transformers::request::{RequestTransformer, ProviderRequestHooks}`
  - `ExecutionPolicy::before_send`
  - `LanguageModelMiddleware::transform_json_body`
- Gateway helper transforms already exist:
  - `to_transcoded_json_response_with_transform(...)`
  - `to_transcoded_json_response_with_response_transform(...)`
  - `to_transcoded_sse_response_with_transform(...)`
- Stream/event middleware already exists:
  - `LanguageModelMiddleware::on_stream_event`
  - `LanguageModelMiddleware::on_stream_end`
  - `LanguageModelMiddleware::on_stream_error`

These hooks are still useful, but they are not sufficient as the primary bridge customization
story because:

- they do not expose source/target bridge context explicitly
- they do not compose with `BridgeReport`
- they do not know whether the planner chose a direct pair bridge or a normalized path
- many of them operate on already-flattened provider JSON or already-encoded SSE
- they encourage route-local glue instead of reusable bridge primitives

## Goals

1. Introduce an explicit bidirectional protocol bridge surface.
2. Keep the normalized internal model as the semantic source of truth.
3. Make lossiness explicit, inspectable, and configurable.
4. Support request, response, and stream customization as first-class bridge hooks.
5. Add a gateway runtime policy layer suitable for embeddable single-tenant or app-level
   gateways.
6. Let users customize conversion behavior without forking pair bridge modules.

## Non-goals

- Do not turn `siumai` into a full Sub2API-style control plane.
- Do not add multi-tenant billing, subscriptions, admin dashboards, or account-pool scheduling to
  this workstream.
- Do not pretend all protocol conversions are exact.
- Do not force all public APIs to become protocol-specific; the normalized family APIs remain the
  recommended application surface.
- Do not make raw provider JSON patching the default bridge extension mechanism.

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
- `siumai-core` owns bridge contracts, loss reporting types, context types, and policy traits
- `siumai` owns planner/facade code plus experimental bridge implementations
- `siumai-extras` owns gateway-ready adapters, route helpers, closure-friendly wrappers, and
  framework integrations

This keeps the core reusable without forcing Axum or route concerns into the protocol/runtime
layer.

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

### D5 - Support customization as a first-class bridge contract

This workstream must explicitly support custom bridge behavior.

Required customization points:

- request pre-bridge transform
- request post-bridge validation
- response bridge transform before target serialization
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

### D5b - Use typed bridge policy traits in core, not a mega override hook

The core bridge contract should prefer small, bridge-aware policy traits with typed context.

Recommended shape:

- `BridgeOptions`
  - mode
  - optional route label
  - request/response/stream hook objects
  - primitive remapper policy
  - lossy handling policy
- typed contexts
  - `RequestBridgeContext`
  - `ResponseBridgeContext`
  - `StreamBridgeContext`
  - `BridgePrimitiveContext`
- policy traits
  - `RequestBridgeHook`
  - `ResponseBridgeHook`
  - `StreamBridgeHook`
  - `BridgePrimitiveRemapper`
  - `BridgeLossPolicy`

This is preferable to a single "override the whole bridge" trait because:

- it stays object-safe and easy to store behind `Arc<dyn ...>`
- it keeps request/response/stream concerns separate
- it avoids generic explosion in public APIs
- it preserves planner and report visibility
- it remains testable without running a full gateway

### D5c - Closure-friendly wrappers belong in `siumai-extras`

The low-level bridge contracts should be trait/policy based.
Gateway helpers may provide closure-friendly adapters on top.

Examples:

- build a `BridgeOptions` from a request closure
- build a `BridgeOptions` from an SSE event mapping closure
- attach route-local labels or strictness defaults in Axum helpers

This gives end users convenient ergonomics without making the core bridge API depend on ad hoc
closure types or framework concerns.

### D5d - Raw JSON patching is an escape hatch, not the primary customization model

We should not recommend "just mutate provider JSON" as the main bridge story.

Why:

- it is protocol-specific by definition
- it usually runs after semantics are already flattened
- it bypasses bridge reports unless the caller re-implements reporting manually
- it does not compose well with direct pair bridges or stream state machines

JSON post-processing may still remain available for gateway experiments, debugging, or
provider-specific one-offs, but it should not be the primary abstraction.

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
- optional route-local bridge options

This layer is intentionally smaller than a full control plane.
It is meant for embeddable gateways inside Rust applications.

Gateway policy must compose bridge customization; it must not invent a second, parallel hook
system.

Current implementation note:

- Axum SSE transcode helpers now enforce keepalive interval and idle-timeout behavior directly from
  `GatewayBridgePolicy`
- Axum JSON/SSE helpers already apply bridge header filtering, warning headers, error passthrough,
  and default bridge-option composition
- request body limits and upstream read limits now have explicit Axum runtime helpers for body and
  JSON reads, instead of being pushed into transcode helpers
- broader non-Axum route/runtime ownership is still future work because the current helper surface
  is intentionally framework-specific

### D7 - Streaming correctness is part of the bridge contract

Stream bridges must be treated as stateful protocol machines, not as line-by-line text rewrites.

Requirements:

- terminal events must be explicit
- finish reasons must survive protocol translation
- content block start/delta/stop ordering must remain valid
- repeated or synthetic events must be deduplicated intentionally
- end-of-stream finalization must be testable without a network

The recent Anthropic SSE fixes fall directly under this rule.

Current implementation note:

- bridge and Axum SSE serialization paths now run an explicit `ensure_stream_end(...)` step before
  target SSE encoding
- target serializers now preserve terminal finish semantics in protocol-native terminal frames
  (`finish_reason`, `stop_reason`, or equivalent) instead of only closing the transport
- Anthropic target serialization now treats block switching as explicit `content_block_stop` /
  `content_block_start` boundaries so interleaved text, thinking, and tool segments remain
  monotonic and protocol-valid
- synthetic finalization only happens for clean EOF without `StreamEnd`, protocol error events, or
  transport errors
- synthetic terminal responses still use `finish_reason = Unknown` when upstream EOF arrives without
  any terminal finish signal, by design

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
pub struct BridgeOptions {
    pub mode: BridgeMode,
    pub route_label: Option<String>,
    pub request_hook: Option<Arc<dyn RequestBridgeHook>>,
    pub response_hook: Option<Arc<dyn ResponseBridgeHook>>,
    pub stream_hook: Option<Arc<dyn StreamBridgeHook>>,
    pub primitive_remapper: Option<Arc<dyn BridgePrimitiveRemapper>>,
    pub loss_policy: Arc<dyn BridgeLossPolicy>,
}

pub struct RequestBridgeContext {
    pub source: Option<BridgeTarget>,
    pub target: BridgeTarget,
    pub mode: BridgeMode,
    pub route_label: Option<String>,
    pub path_label: Option<String>,
}

pub trait RequestBridgeHook: Send + Sync {
    fn transform_request(
        &self,
        ctx: &RequestBridgeContext,
        request: &mut ChatRequest,
        report: &mut BridgeReport,
    ) -> Result<(), LlmError>;

    fn validate_json(
        &self,
        ctx: &RequestBridgeContext,
        body: &serde_json::Value,
        report: &mut BridgeReport,
    ) -> Result<(), LlmError>;
}

pub trait BridgePrimitiveRemapper: Send + Sync {
    fn remap_tool_name(
        &self,
        ctx: &BridgePrimitiveContext,
        name: &str,
    ) -> Option<String>;

    fn remap_tool_call_id(
        &self,
        ctx: &BridgePrimitiveContext,
        id: &str,
    ) -> Option<String>;
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
  - bridge contracts, reports, contexts, and policy traits
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

1. Define bridge contracts, typed contexts, and default loss policy.
2. Refactor request bridge code into planner + primitives + pair modules.
3. Stabilize explicit normalized request -> target protocol request bridges.
4. Stabilize non-streaming response bridges.
5. Stabilize streaming bridges and finalization semantics.
6. Add bridge customization policies and primitive remappers.
7. Add gateway runtime policy objects and route helpers on top of bridge customization.
8. Add examples, fixtures, and migration notes.

## Success criteria

This workstream is successful when:

- a user can explicitly bridge request, response, or stream shapes across protocols
- lossy behavior is reported and configurable
- gateway helpers are built on top of the bridge layer, not vice versa
- protocol semantics are preserved by protocol-owned state machines
- custom hooks are supported through stable typed options instead of patching internal converter code
- adding one new bridge route does not require writing a new layer of pairwise glue
