# Protocol Bridge + Gateway Runtime - Customization Boundary

Last updated: 2026-03-22

This note makes one architectural decision explicit:

- user-defined bridge behavior is supported
- user-defined bridge behavior is not the same thing as exposing a runtime `N x M` parser /
  serializer registry

The current repository already supports custom conversion. The supported shape is a small set of
typed lifecycle traits and policies attached to `BridgeOptions` instead of one giant
"override everything" hook.

## Recommended extension order

Pick the smallest layer that matches the change you need.

| Need | Recommended surface | Why |
| --- | --- | --- |
| Coordinate request / response / stream customization in one object | `BridgeCustomization` + `BridgeOptions::with_customization(...)` | Lowest-glue path for reusable bridge policy bundles |
| Customize one route or caller locally without declaring a dedicated struct | `siumai-extras::bridge::ClosureBridgeCustomization` | Closure-friendly ergonomics built on the same typed bridge contract |
| Rename tool names or tool-call IDs across request / response / stream paths | `BridgePrimitiveRemapper` | Reusable, bridge-aware, and cheap to test |
| Reject, warn, or continue on lossy routes | `BridgeLossPolicy` | Keeps strictness logic tied to `BridgeReport` |
| Rewrite normalized requests before target serialization | `RequestBridgeHook::transform_request` | Mutates semantics before target JSON exists |
| Apply target-specific JSON tweaks after semantic mapping | `RequestBridgeHook::transform_json` and `validate_json` | Last-mile wire adjustment without forking pair bridges |
| Rewrite normalized responses before target serialization | `ResponseBridgeHook::transform_response` | Response-side semantic filtering / enrichment |
| Rewrite streaming events before target SSE encoding | `StreamBridgeHook::map_event` | Stream-aware mutation without patching encoded frames |
| Override route defaults without rebuilding the full bridge config | `BridgeOptionsOverride` | Gateway-friendly partial override |

That is the intended public customization model.

## What is intentionally supported today

### 1) Typed lifecycle traits

The bridge surface now exposes:

- `BridgeCustomization`
- `RequestBridgeHook`
- `ResponseBridgeHook`
- `StreamBridgeHook`
- `BridgePrimitiveRemapper`
- `BridgeLossPolicy`
- `BridgeOptions`
- `BridgeOptionsOverride`

These hooks all receive typed bridge context:

- source protocol
- target protocol
- request phase for request-side hooks (`NormalizeSource` vs `SerializeTarget`)
- bridge mode
- route label
- planner / path label

That is important because the same customization can behave differently on:

- same-protocol replay
- cross-protocol normalized bridges
- selected direct pair bridges

`BridgeCustomization` is an ergonomic bundle over those lower-level interfaces.

It does not replace them. It exists so applications can attach one reusable object when the same
route policy needs to:

- remap tool primitives
- mutate normalized requests
- rewrite responses
- inspect or reject lossy bridges
- transform stream events

For application and gateway-local ergonomics, `siumai-extras` also exposes closure-friendly
wrappers that implement the same bridge contract for you. The canonical import path is
`siumai-extras::bridge`, and gateway helpers re-export the same adapters from
`siumai-extras::server::axum`:

- `ClosureRequestBridgeHook`
- `ClosureBridgeCustomization`
- `request_bridge_hook(...)`
- `ClosureResponseBridgeHook`
- `ClosureStreamBridgeHook`
- `response_bridge_hook(...)`
- `stream_bridge_hook(...)`
- `ClosurePrimitiveRemapper`

### 2) Protocol-source request normalization stays explicit

Inbound protocol-native requests should still normalize through protocol-owned parsers first:

- `bridge_anthropic_messages_json_to_chat_request(...)`
- `bridge_gemini_generate_content_json_to_chat_request(...)`
- `bridge_openai_responses_json_to_chat_request(...)`
- `bridge_openai_chat_completions_json_to_chat_request(...)`

If an application wants inbound customization, the recommended entry points are the explicit
`with_options` variants:

- `bridge_anthropic_messages_json_to_chat_request_with_options(...)`
- `bridge_gemini_generate_content_json_to_chat_request_with_options(...)`
- `bridge_openai_responses_json_to_chat_request_with_options(...)`
- `bridge_openai_chat_completions_json_to_chat_request_with_options(...)`

Those entry points keep parser ownership in protocol code, then apply the same typed bridge
customization surface to the normalized `ChatRequest`.

The intended composition is:

1. protocol-native JSON -> protocol-owned parse -> `ChatRequest`
2. bridge-owned post-normalize remap / hook / loss-policy application
3. unified runtime execution
4. response / stream bridge to the desired target view

This keeps protocol parsing rules testable and prevents route-local parser forks from becoming a
second bridge system.

### 3) Direct pair bridges are in-tree architectural choices, not end-user plugins

The repository already uses a hybrid strategy:

- normalized backbone for most paths
- selected direct pair bridges only where loss would otherwise be material

Current examples:

- Anthropic Messages -> OpenAI Responses
- OpenAI Responses -> Anthropic Messages

These direct pair bridges are intentionally curated in-tree. End users should not have to fork them
for small custom behavior, because the typed hooks and remappers are supposed to cover that space.

### 4) Direct pair hosted-tool translation is curated, but user-extensible

The direct Anthropic Messages <-> OpenAI Responses bridges now carry in-tree hosted-tool
translation rules for the highest-value semantic pairs instead of scattering those mappings across
one large pair file.

Current curated translations include:

- Anthropic `web_search_20250305` -> OpenAI `openai.web_search`
- Anthropic `code_execution_20250522` / `code_execution_20250825` -> OpenAI
  `openai.code_interpreter`
- OpenAI `web_search` / `web_search_preview` -> Anthropic `web_search_20250305`
- OpenAI `code_interpreter` -> Anthropic `code_execution_20250825`
- OpenAI `computer_use` -> Anthropic `computer_20250124`
- OpenAI `mcp` -> Anthropic `mcp_servers` through a dedicated helper, not as a generic
  provider-defined tool rewrite

That split is intentional:

- repository-owned direct bridges keep the common hosted-tool mappings in one audited place
- users do not need to copy those tables into route-local glue
- uncommon or application-specific hosted-tool compatibility is handled by a narrow customization
  surface instead of introducing a runtime `N x M` plugin mesh

## What is intentionally not the default model

### 1) No whole-parser override trait

We are intentionally not adding one trait like:

```rust
trait CustomBridgeEverything {
    fn parse_source(&self, json: Value) -> ChatRequest;
    fn serialize_target(&self, request: &ChatRequest) -> Value;
}
```

That would:

- recreate pairwise glue at the parser boundary
- hide `BridgeReport` decisions inside route-local code
- make direct-pair and normalized paths harder to reason about
- push protocol ownership away from protocol crates

### 2) No runtime `N x M` bridge plugin matrix

Adding a new first-class protocol target still belongs to repository code, not end-user runtime
registration.

That keeps:

- protocol wire contracts owned by protocol crates
- target inspection logic owned by bridge code
- serializer state machines testable in one place

### 3) Raw JSON / SSE patching is an escape hatch

Route-local JSON or frame mutation helpers still exist, but they are not the primary extension
story.

Use them only when the requirement is truly wire-format-specific.

If the change is semantic, prefer the typed bridge hooks first.

## Recommended decision tree

### Case A - "I only need naming policy"

Use `BridgePrimitiveRemapper`.

Examples:

- tenant-prefixed tool names
- deterministic tool-call ID rewrite
- route-specific function naming policy

### Case B - "I need to reject some lossy bridges but not all"

Use `BridgeLossPolicy`.

Examples:

- reject dropped reasoning on one route
- allow source metadata loss on another route
- keep `BestEffort` globally but tighten a regulated endpoint

### Case C - "I need to change normalized semantics"

Use:

- `RequestBridgeHook::transform_request`
- `ResponseBridgeHook::transform_response`
- `StreamBridgeHook::map_event`

Examples:

- inject route-owned system guidance
- redact response content before target serialization
- rewrite streaming deltas for auditing or masking

### Case D - "I need to tweak one target JSON field"

Use `RequestBridgeHook::transform_json` plus `validate_json`.

This is the narrow place where target-specific mutation is acceptable without turning the whole
bridge into raw JSON patch code.

### Case E - "I need a custom hosted-tool rewrite across protocols"

Use `ProviderToolRewriteCustomization`.

This is the preferred escape hatch when:

- the direct pair bridge does not already know your hosted tool
- you want to translate one provider-defined tool into another before target serialization
- you want to map arguments semantically instead of patching the final target JSON

Minimal pattern:

```rust
use std::sync::Arc;

use serde_json::json;
use siumai::experimental::bridge::{
    BridgeMode, BridgeOptions, BridgeTarget, ProviderToolRewriteCustomization,
    ProviderToolRewriteRule, bridge_chat_request_to_openai_responses_json_with_options,
};

let customization = ProviderToolRewriteCustomization::new().with_rule(
    ProviderToolRewriteRule::new("anthropic.web_fetch_20250910", "openai.web_search")
        .with_args_mapper(Arc::new(|_ctx, tool, _report| {
            let allowed_domains = tool
                .args
                .get("allowedDomains")
                .cloned()
                .unwrap_or_else(|| json!([]));

            json!({
                "filters": {
                    "allowedDomains": allowed_domains,
                }
            })
        })),
);

let bridged = bridge_chat_request_to_openai_responses_json_with_options(
    &request,
    Some(BridgeTarget::AnthropicMessages),
    BridgeOptions::new(BridgeMode::BestEffort).with_customization(Arc::new(customization)),
)?;
```

Recommended rule:

- if the change is "translate hosted tool semantics", prefer `ProviderToolRewriteCustomization`
- if the change is "adjust final wire fields after semantics are already correct", use the JSON
  overlay hooks
- do not start with raw JSON patching for hosted-tool conversion

## Minimal shape for applications

Reusable application-level policy bundle:

```rust
let options = BridgeOptions::new(BridgeMode::BestEffort)
    .with_route_label("gateway.responses")
    .with_customization(Arc::new(MyBridgeCustomization));
```

Route-local Axum helper wiring without declaring a dedicated customization struct:

```rust
use siumai_extras::bridge::ClosureBridgeCustomization;

let opts = TranscodeSseOptions::default().with_bridge_customization(Arc::new(
    ClosureBridgeCustomization::default()
        .with_stream(|ctx, event| {
            assert_eq!(ctx.target, BridgeTarget::OpenAiResponses);
            vec![event]
        })
        .with_stream_loss_action(|_ctx, report| {
            if report.is_rejected() {
                BridgeLossAction::Reject
            } else {
                BridgeLossAction::Continue
            }
        }),
));
```

Use the struct-based `BridgeCustomization` path when the policy is reusable across routes or crates.
Use the closure-friendly wrappers when the customization is local to one gateway adapter, test, or
application entry point.

Gateway routes that only need partial overrides should prefer `BridgeOptionsOverride`.

If only one phase needs customization, keep using the smaller dedicated trait instead of forcing a
bundle object.

## Trait vs closure decision

Use the struct-based `BridgeCustomization` path when:

- the same policy bundle should be reused across routes, tests, or crates
- one policy needs to coordinate request mutation, target JSON overlays, validation, and primitive
  remapping together
- the route should expose a named architectural policy instead of anonymous inline closures

Use closure-friendly helpers from `siumai-extras::bridge` when:

- the customization is route-local
- the policy is only needed inside one gateway adapter, smoke test, or example
- a dedicated named type would add ceremony without adding reuse

That yields a simple rule:

- reusable bridge policy -> `BridgeCustomization`
- route-local bridge policy -> `ClosureBridgeCustomization`
- one narrow lifecycle phase only -> dedicated hook trait (`RequestBridgeHook`,
  `ResponseBridgeHook`, `StreamBridgeHook`, `BridgePrimitiveRemapper`, or `BridgeLossPolicy`)

## Protocol-target JSON overlay recipe

When the target protocol needs one last wire-specific adjustment, keep the semantic and wire steps
separate:

1. rewrite normalized semantics in `transform_request(...)` when the meaning of the request is
   changing
2. add target-only JSON overlays in `transform_request_json(...)`
3. assert the final wire contract in `validate_request_json(...)`
4. keep parser ownership and baseline target serialization in protocol crates

That keeps bridge custom behavior auditable without reintroducing raw JSON patch glue as the main
API.

Minimal pattern:

```rust
let options = BridgeOptions::new(BridgeMode::BestEffort)
    .with_route_label("gateway.responses")
    .with_customization(Arc::new(MyBridgeCustomization));
```

Inside `MyBridgeCustomization`:

- branch on `RequestBridgeContext.phase`
- branch on `RequestBridgeContext.target`
- mutate `ChatRequest` before serialization only for semantic changes
- mutate `serde_json::Value` after serialization only for target-wire overlays
- validate the final body before returning it to the caller

## Gateway reuse path

Gateway adapters reuse the same bridge customization objects instead of defining a second gateway
hook system.

That means the same `ProviderToolRewriteCustomization` or custom `BridgeCustomization` object can
be attached through:

- `BridgeOptions::with_customization(...)`
- `BridgeOptionsOverride::with_customization(...)`
- `GatewayBridgePolicy::with_customization(...)`
- `NormalizeRequestOptions::with_bridge_customization(...)`
- `TranscodeJsonOptions::with_bridge_customization(...)`
- `TranscodeSseOptions::with_bridge_customization(...)`

That is the intended layering:

1. define semantic bridge policy once
2. reuse it in direct bridge code or gateway route helpers
3. avoid separate gateway-only hosted-tool rewrite logic

## Runnable references

- Pure bridge request customization:
  - `siumai/examples/06-extensibility/bridge-customization.rs`
- Gateway request-normalization + backend execution with hosted-tool rewrite reuse:
  - `siumai-extras/examples/anthropic-to-openai-responses-gateway.rs`
- Route-local gateway closure customization:
  - `siumai-extras/examples/gateway-custom-transform.rs`

## Current conclusion

The repository already supports user-defined conversion, but it supports it in a constrained,
typed, bridge-aware way:

- yes to hook traits and policy objects
- yes to a single bundled `BridgeCustomization` object when one policy spans multiple phases
- yes to normalized post-parse transforms
- yes to reusable primitive remappers
- no to a catch-all parser override trait
- no to a runtime `N x M` bridge plugin mesh

That boundary is deliberate. It keeps the bridge reusable without turning it into another layer of
hard-to-audit glue.
