# 2026-05-18 - AIPC-050 OpenAI Responses Tool Stream Parts

Started AIPC-050 with an OpenAI Responses vertical slice.

Changes:

- Updated `siumai-protocol-openai/tests/responses_sse_feature_surface_test.rs` so public feature
  surface tests serialize stable `ChatStreamEvent::Part` tool call/result parts.
- Added a boundary test in `siumai-protocol-openai/tests/openai_compat_boundary_test.rs` to keep
  public OpenAI Responses tool stream part tests from regressing to `Custom("openai:*")` inputs.
- Left converter-level `ChatStreamEvent::Custom` coverage in place as an explicit compatibility
  path, not the stable public stream-part path.

Reference alignment:

- AI SDK stream semantics treat `tool-call` and `tool-result` as stable stream parts. Provider wire
  details remain an adapter concern.

Validation:

- `cargo fmt -p siumai-protocol-openai --check`
- `cargo nextest run -p siumai-protocol-openai --features openai-standard,openai-responses --test responses_sse_feature_surface_test --no-fail-fast`
- `cargo nextest run -p siumai-protocol-openai responses_feature_surface_uses_stable_parts_for_tool_stream_parts --no-fail-fast`
- `cargo nextest run -p siumai-protocol-openai --all-features --no-fail-fast`

Next task:

- Continue AIPC-050 by auditing Anthropic/Gemini and `siumai-extras` gateway stream tests for
  stable semantics still asserted through provider custom/raw replay data.
