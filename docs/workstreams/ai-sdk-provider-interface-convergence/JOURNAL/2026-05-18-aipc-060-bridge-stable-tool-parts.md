# AIPC-060 Bridge Stable Tool Parts

Date: 2026-05-18
Task: AIPC-060
Status: In progress

## Summary

The first AIPC-060 slice moved bridge/gateway evidence closer to stable-part-first behavior.
`siumai-bridge` now has a stream regression test that feeds stable provider-executed tool input,
tool call, and tool result parts into the OpenAI Responses bridge and verifies the serialized SSE
output item frames. This proves the bridge path does not depend on provider custom stream payloads
for the primary provider-tool semantics.

## Seam Cleanup

`siumai-extras` already depends on `siumai-bridge` directly. The Axum OpenAI Responses SSE helper
now imports `OpenAiResponsesStreamPartsBridge` from `siumai_bridge::stream` instead of through
`siumai::experimental::streaming`. The facade re-export remains available for advanced users, but
runtime gateway code should use the bridge crate seam directly.

## Gates

- `cargo fmt -p siumai-bridge -p siumai-extras -- --check`
- `cargo nextest run -p siumai-bridge --features openai,google openai_responses_stream_bridge_maps_stable_provider_tool_parts_to_output_items --no-fail-fast`
- `cargo nextest run -p siumai-extras --features server,openai openai_responses_gateway_parts_adapter_uses_bridge_stream_seam_directly --no-fail-fast`
- `cargo nextest run -p siumai-bridge --all-features --no-fail-fast`
- `cargo nextest run -p siumai-extras --features server,openai,anthropic,google --test gateway_axum_smoke_test --no-fail-fast`
- `cargo nextest run -p siumai-extras --features server,openai --test bridge_architecture_boundary_test --no-fail-fast`

## Next

Audit whether the direct extras SSE transcode helper still needs a stable provider-tool stream part
test. If not, close AIPC-060 and move the program to provider package parity.
