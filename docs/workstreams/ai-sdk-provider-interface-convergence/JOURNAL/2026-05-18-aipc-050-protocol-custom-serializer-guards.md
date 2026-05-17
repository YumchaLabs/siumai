# 2026-05-18 - AIPC-050 Protocol Custom Serializer Guards

Continued AIPC-050 with Anthropic and Gemini serializer boundary guards.

Changes:

- Added `anthropic_streaming_serializer_custom_inputs_are_compat_or_provider_native_only`.
- Added `gemini_streaming_serializer_custom_inputs_are_compat_or_provider_native_only`.
- The guards allow `ChatStreamEvent::Custom` serializer inputs only when the test name explicitly
  marks the case as V3 compatibility, provider-native custom behavior, or compatibility behavior.

Rationale:

- Anthropic and Gemini already have stable `ChatStreamEvent::Part` serializer coverage.
- Remaining custom-event serializer tests are intentional compatibility/provider-native coverage and
  should not become the default test shape for stable AI SDK stream semantics.

Validation:

- `cargo fmt -p siumai-protocol-anthropic -p siumai-protocol-gemini --check`
- `cargo nextest run -p siumai-protocol-anthropic --all-features anthropic_streaming_serializer_custom_inputs_are_compat_or_provider_native_only --no-fail-fast`
- `cargo nextest run -p siumai-protocol-gemini --all-features gemini_streaming_serializer_custom_inputs_are_compat_or_provider_native_only --no-fail-fast`
- `cargo nextest run -p siumai-protocol-anthropic --all-features --no-fail-fast`
- `cargo nextest run -p siumai-protocol-gemini --all-features --no-fail-fast`

Next task:

- Decide whether AIPC-050 is now sufficient to close, or whether extras object/tool-loop loose
  custom parsers should be split into a smaller follow-up compatibility-boundary task.
