# Anthropic Extended Usage Roundtrip

Last updated: 2026-04-14

## Goal

Close the public-surface regression behind GitHub issue `#17`:
`AnthropicEventConverter loses the extended usage fields`.

The protocol crate already had internal serializer coverage, but the public `siumai` facade still
needed one top-level regression that proves the public Anthropic converter preserves extended usage
metadata across an actual encode/decode roundtrip.

Reference:

- `https://github.com/YumchaLabs/siumai/issues/17`

## What Changed

- Added a new top-level regression test:
  `siumai/tests/anthropic_extended_usage_fields_roundtrip_test.rs`
- The test exercises the public
  `siumai::protocol::anthropic::streaming::AnthropicEventConverter` directly instead of only the
  lower-level protocol crate internals.
- The roundtrip now explicitly pins the three fields called out by the issue:
  - `cache_read_input_tokens`
  - `service_tier`
  - `server_tool_use`
- The test also keeps `cache_creation_input_tokens` in scope because the serializer rebuilds the
  full Anthropic `usage` object rather than preserving only one isolated field.

## Why This Matters

- AI SDK treats Anthropic extended usage as stable provider metadata, not as disposable transport
  detail.
- The public Rust surface should therefore guarantee those fields survive:
  - `ChatResponse.provider_metadata["anthropic"].usage`
  - SSE serialization through `AnthropicEventConverter`
  - SSE decoding back into a typed `ChatResponse`
  - `AnthropicChatResponseExt::anthropic_metadata()`

Without this top-level regression, the protocol crate could stay green while the public facade
still regressed on the real issue boundary.

## Validation

- `cargo nextest run -p siumai --all-features anthropic_public_roundtrip_preserves_extended_usage_fields`
