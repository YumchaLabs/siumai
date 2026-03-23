# Protocol Bridge + Gateway Runtime - Route Recipes

Last updated: 2026-03-23

This note turns the current bridge/gateway surface into a small set of route shapes that are
worth copying directly.

The goal is to avoid ad hoc pairwise glue and keep routes aligned with the explicit bridge stack:

1. policy-aware body read
2. explicit request normalization or upstream read
3. unified runtime execution or buffered upstream handling
4. explicit JSON/SSE transcode

## Recipe 1: Provider-native ingress -> unified runtime -> target JSON/SSE

Use this shape when the downstream client sends provider-native request JSON and your route should
execute one normalized `ChatRequest`.

Recommended steps:

- read the downstream request with:
  - `read_request_json_with_policy(...)`
- normalize the source request with:
  - `normalize_request_json_with_options(...)`
- apply route defaults through:
  - `GatewayBridgePolicy`
- use route-local strictness or custom bridge behavior through:
  - `NormalizeRequestOptions`
  - `BridgeOptionsOverride`
- execute the normalized `ChatRequest`
- transcode the unified response/stream with:
  - `to_transcoded_json_response(...)`
  - `to_transcoded_sse_response(...)`

Use this recipe when:

- one ingress protocol should feed one unified model/runtime path
- you want source normalization to stay explicit and testable
- request-side customization should remain typed instead of patching raw JSON

Reference examples:

- `siumai-extras/examples/anthropic-to-openai-responses-gateway.rs`
- `siumai-extras/examples/openai-responses-to-anthropic-gateway.rs`

Reference tests:

- `siumai-extras/tests/request_normalize_smoke_test.rs`
- `siumai-extras/tests/gateway_request_ingress_axum_test.rs`
- `siumai-extras/tests/gateway_request_ingress_sse_axum_test.rs`

## Recipe 2: Buffered upstream proxy/runtime route -> target JSON

Use this shape when the route first reads a buffered upstream provider response body and then
re-emits a normalized or summarized downstream JSON response.

Recommended steps:

- read upstream bytes or JSON with:
  - `read_upstream_body_with_policy(...)`
  - `read_upstream_json_with_policy(...)`
- let `GatewayBridgePolicy` own:
  - `upstream_read_limit_bytes`
  - runtime error masking via `with_passthrough_runtime_errors(false)`
- after the upstream read succeeds, emit the downstream provider view with:
  - `to_transcoded_json_response(...)`

Use this recipe when:

- the gateway is buffering an upstream response before returning JSON
- you need one place to enforce upstream size limits and error masking

Reference tests:

- `siumai-extras/tests/gateway_runtime_proxy_axum_test.rs`

## Recipe 3: Buffered upstream or unified runtime -> target SSE

Use this shape when the route should emit a downstream SSE body.

Recommended steps:

- if the route starts from a unified stream, call:
  - `to_transcoded_sse_response(...)`
- if the route starts from a provider-native ingress request, normalize first, execute, then
  transcode the resulting `ChatStream`
- if the route is cross-protocol and you need inspected strict rejection or custom loss policy,
  declare the upstream protocol with:
  - `TranscodeSseOptions::with_bridge_source(...)`
- let `GatewayBridgePolicy` own runtime timers:
  - `with_keepalive_interval(...)`
  - `with_stream_idle_timeout(...)`

Use this recipe when:

- one upstream stream needs multiple downstream protocol views
- the route should reject lossy cross-protocol streaming in `Strict` mode
- keepalive and idle-timeout behavior must stay policy-driven instead of route-local

Reference examples:

- `siumai-extras/examples/openai-responses-gateway.rs`
- `siumai-extras/examples/gateway-loss-policy.rs`

Reference tests:

- `siumai-extras/tests/gateway_axum_smoke_test.rs`
- `siumai-extras/tests/gateway_request_ingress_sse_axum_test.rs`
- `siumai-extras/tests/gateway_runtime_proxy_axum_test.rs`

## Recipe 4: Hosted-tool compatibility on ingress

If the request-side problem is provider-hosted tool compatibility, do not patch the final wire
JSON first.

Prefer:

- `ProviderToolRewriteCustomization`
- attach it through:
  - `GatewayBridgePolicy::with_customization(...)`
  - or `NormalizeRequestOptions::with_bridge_customization(...)`

Use this recipe when:

- Anthropic hosted tools need to become OpenAI hosted tools before downstream execution
- the route should preserve typed bridge context and `BridgeReport`

Reference example:

- `siumai-extras/examples/anthropic-to-openai-responses-gateway.rs`

Reference tests:

- `siumai-extras/tests/request_normalize_smoke_test.rs`
- `siumai-extras/tests/gateway_request_ingress_axum_test.rs`
- `siumai-extras/tests/gateway_request_ingress_sse_axum_test.rs`

## Anti-patterns

Avoid starting with these patterns unless the requirement is truly wire-specific:

- open-coded `to_bytes(...)` for downstream or upstream bodies
- route-local raw JSON patching before or after bridge/transcode
- route-local `if strict { ... }` branching that duplicates `BridgeLossPolicy`
- adding new pairwise route glue before trying the normalized backbone plus selected direct bridges

Those paths make it harder to keep:

- `BridgeReport`
- route labels and path labels
- loss-policy behavior
- runtime limits/timeouts

aligned across the gateway surface.

## Current tested route matrix

The route patterns above are currently backed by these focused tests:

- request normalizers:
  - `siumai-extras/tests/request_normalize_smoke_test.rs`
- ingress JSON:
  - `siumai-extras/tests/gateway_request_ingress_axum_test.rs`
- ingress SSE:
  - `siumai-extras/tests/gateway_request_ingress_sse_axum_test.rs`
- runtime/proxy:
  - `siumai-extras/tests/gateway_runtime_proxy_axum_test.rs`
- egress and cross-protocol target views:
  - `siumai-extras/tests/gateway_axum_smoke_test.rs`

This is the recommended baseline for new gateway routes in this repository.
