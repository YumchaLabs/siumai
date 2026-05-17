# 2026-05-18 - AIPC-050 Gateway Tool Stream Parts

Continued AIPC-050 with an extras gateway assertion slice.

Changes:

- Tightened `siumai-extras/tests/gateway_axum_smoke_test.rs` from a typed-or-custom tool part helper
  to a stable-part-only helper.
- Added `gateway_smoke_asserts_stable_tool_stream_parts` so the gateway smoke test cannot accept
  `ChatStreamEvent::Custom` payloads as proof of stable tool call/result behavior.
- Kept server/gateway production compatibility paths unchanged.

Validation:

- `cargo fmt -p siumai-extras --check`
- `cargo nextest run -p siumai-extras --features server,openai,anthropic gateway_smoke_asserts_stable_tool_stream_parts --no-fail-fast`
- `cargo nextest run -p siumai-extras --features server,openai,anthropic gateway_route_smoke_transcodes_anthropic_fixture_to_openai_sse --no-fail-fast`
- `cargo nextest run -p siumai-extras --features server,openai,anthropic,google --test gateway_axum_smoke_test --no-fail-fast`

Next task:

- Continue AIPC-050 by auditing protocol serializer unit tests for stable stream parts still covered
  only through provider custom/raw inputs.
