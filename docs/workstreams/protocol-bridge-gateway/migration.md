# Protocol Bridge + Gateway Runtime - Migration Note

Last updated: 2026-03-20

This note explains how gateway code should migrate toward the explicit bridge surface now that
`BridgeOptions`, `BridgeOptionsOverride`, and `GatewayBridgePolicy` exist.

If you want concrete route shapes instead of migration principles, pair this note with:

- `docs/workstreams/protocol-bridge-gateway/route-recipes.md`

## Recommended migration path

### 1) Route-local JSON patching is no longer the primary story

Older gateway code often started from one of these patterns:

- mutate provider JSON before send
- patch serialized JSON after `ChatResponse` encoding
- patch SSE frames after provider-native serialization
- scatter per-route `strict` vs `best-effort` decisions in handlers

Those escape hatches still exist, but they should not be the default architecture because they:

- lose bridge-aware context (`source`, `target`, path label, route label)
- do not compose naturally with `BridgeReport`
- encourage pairwise route glue instead of reusable policies
- make strictness and lossy handling hard to audit

### 2) Prefer the explicit bridge stack

Use the smallest typed layer that matches the change you actually need:

- request normalization:
  - `bridge_anthropic_messages_json_to_chat_request(...)`
  - `bridge_openai_responses_json_to_chat_request(...)`
  - `bridge_openai_chat_completions_json_to_chat_request(...)`
- route/runtime defaults:
  - `GatewayBridgePolicy`
- route-level partial overrides:
  - `BridgeOptionsOverride`
- semantic customization:
  - `RequestBridgeHook`
  - `ResponseBridgeHook`
  - `StreamBridgeHook`
  - `BridgePrimitiveRemapper`
  - `BridgeLossPolicy`
- target serialization:
  - `to_transcoded_json_response(...)`
  - `to_transcoded_sse_response(...)`

### 3) Migrate by intent, not by helper name

If the old route exists only to pick a target protocol:

- move to `to_transcoded_json_response(...)` or `to_transcoded_sse_response(...)`
- set route defaults through `GatewayBridgePolicy`

If the old route mutates normalized semantics:

- move the logic into typed bridge hooks or a primitive remapper
- keep raw JSON mutation only for wire-specific last-mile quirks

If the old route accepts protocol-native downstream bodies:

- normalize first into `ChatRequest`
- then apply any additional normalized request transform
- then execute the unified request

If the old route needs custom “reject vs continue” behavior on lossy bridges:

- implement `BridgeLossPolicy`
- install it through `BridgeOptions` or `BridgeOptionsOverride`

## Current helper guidance

### JSON gateway helpers

`siumai-extras` JSON transcode helpers already run the explicit response bridge and therefore
consume:

- `BridgeMode`
- `BridgeOptions`
- `BridgeOptionsOverride`
- `BridgeLossPolicy`
- `GatewayBridgePolicy`

That means route-level strictness, custom loss policy, typed response hooks, and primitive
remappers can all stay on the bridge path instead of becoming route glue.

Reference example:

- `siumai-extras/examples/gateway-loss-policy.rs`

### SSE gateway helpers

Axum SSE transcode helpers already consume:

- `GatewayBridgePolicy`
- `BridgeMode`
- `BridgeOptions`
- `BridgeOptionsOverride`
- `BridgeLossPolicy`
- typed stream hooks
- primitive remappers
- keepalive / idle timeout runtime policy
- bridge target / mode / decision headers
- warning counter headers

They now also run the same inspected stream-loss decision loop as the core
`bridge_chat_stream_to_*_sse_with_options(...)` entry points when the caller declares the upstream
protocol with `TranscodeSseOptions::with_bridge_source(...)`.

Today this means:

- if the route is same-protocol or the upstream source is intentionally unknown,
  `to_transcoded_sse_response(...)` can still be used without declaring a source
- if you need inspected strict rejection / custom `BridgeLossPolicy` decisions on cross-protocol
  streaming routes, set `with_bridge_source(...)` on the SSE transcode options and stay on the
  convenience helper path

Reference example:

- `siumai-extras/examples/gateway-loss-policy.rs`

## What should remain as escape hatches

These APIs are still valid, but should be treated as secondary tools:

- `to_transcoded_json_response_with_transform(...)`
- `to_transcoded_json_response_with_response_transform(...)`
- `to_transcoded_sse_response_with_transform(...)`

Use them when:

- the tweak is intentionally wire-format-specific
- the route is experimental and not yet worth a reusable policy object
- you are debugging bridge output and want a temporary instrumented route

Do not start with them when the real requirement is:

- typed request/response/stream semantic mutation
- route-wide strictness control
- reusable tool name / tool call id policy
- reusable lossy handling policy

## Migration checklist

- Move source-protocol body parsing into explicit request normalizers.
- Move route defaults into `GatewayBridgePolicy`.
- Move semantic rewrites into typed bridge hooks or primitive remappers.
- Move lossy handling into `BridgeLossPolicy`.
- Keep raw JSON/frame transforms only as wire-level escape hatches.
- Prefer one normalized runtime path plus selected direct bridges, not ad hoc pairwise route glue.
