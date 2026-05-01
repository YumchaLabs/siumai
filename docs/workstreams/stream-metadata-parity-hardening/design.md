# Stream Metadata Parity Hardening - Design

Last updated: 2026-05-01

## Problem

The ongoing AI SDK alignment work exposed a second cluster of parity gaps around stream semantics
and provider metadata extraction:

- the shared stream factory now replays legacy textual shadow deltas for stable runtime
  `TextDelta` / `ReasoningDelta` parts, but that replay could double-emit
  `ContentDelta` / `ThinkingDelta` when a converter already emitted the legacy lane itself
- the Perplexity OpenAI-compatible metadata helper normalized hosted-search metadata into
  `providerMetadata.perplexity`, but it still dropped `usage.reasoningTokens`
- Gemini / Vertex reasoning streams emitted stable runtime parts and `ThinkingDelta`, but they did
  not also expose AI SDK-style custom `reasoning-*` events on the public stream surface
- after adding those Gemini custom reasoning events, the Gemini GenerateContent serializer could
  see both the typed `Part` lane and the mirrored `Custom` lane during bridge round-trips and
  serialize the same reasoning delta twice
- the shared OpenAI-compatible metadata helper normalized DeepSeek prompt-cache accounting into
  stable `Usage`, but did not mirror AI SDK's `providerMetadata.deepseek.promptCacheHitTokens` /
  `promptCacheMissTokens` response metadata fields

These are all parity regressions of the same general kind: the stable runtime lane is present, but
the compatibility lane or metadata surface is still incomplete.

## Design

### 1. Make textual shadow replay idempotent

The shared stream factory keeps the canonical compatibility rule:

- stable textual parts remain the primary runtime representation
- legacy textual delta consumers still receive best-effort `ContentDelta` / `ThinkingDelta`

But the replay step now first checks whether the converter batch already contains those legacy
delta events. If the legacy lane already exists, the factory does not synthesize another copy.

This keeps the compatibility layer safe for mixed converters that emit both lanes during
transitional refactors.

### 2. Normalize Perplexity hosted-search usage completely

Perplexity response metadata is intentionally projected into an AI SDK-shaped typed provider
namespace. That normalization must include the full hosted-search usage subset, not only the older
`citationTokens` and `numSearchQueries` fields.

The metadata extractor therefore now also maps:

- `usage.reasoning_tokens`
- `usage.reasoningTokens`

into:

- `providerMetadata.perplexity.usage.reasoningTokens`

This keeps response metadata parity consistent between raw compat JSON, typed helpers, and public
path tests.

### 3. Expose Gemini reasoning on both stable lanes

Gemini / Vertex reasoning chunks now emit:

- stable runtime `Part(ChatStreamPart::{ReasoningStart,ReasoningDelta,ReasoningEnd})`
- legacy-compatible `ThinkingDelta`
- AI SDK-style custom `reasoning-start` / `reasoning-delta` / `reasoning-end`

The public stream surface therefore matches the practical AI SDK expectation:

- part-aware consumers can use the semantic lane
- legacy reasoning-text consumers can still read `ThinkingDelta`
- fixture/public-path tests that inspect AI SDK custom event payloads continue to pass

### 4. Deduplicate mixed `Part + Custom` reasoning during Gemini re-serialization

The Gemini serializer remains responsible for provider wire output, so duplicate suppression is
applied there instead of weakening the runtime event surface.

Specifically:

- when a typed Gemini `ReasoningDelta` part is mirrored into the Gemini custom lane for
  serialization, the serializer records that one duplicate custom delta may follow
- if the next custom `reasoning-delta` matches the pending typed delta exactly, it is suppressed
- the suppression window is cleared as soon as the pending reasoning chunk is flushed

This keeps bridge round-trips stable without regressing the richer runtime event model.

### 5. Mirror DeepSeek prompt-cache usage into provider metadata

DeepSeek follows the OpenAI-compatible Chat Completions shape but exposes prompt-cache accounting as
top-level usage fields. Stable Siumai usage conversion already consumes
`usage.prompt_cache_hit_tokens` / `usage.prompt_cache_miss_tokens`; the AI SDK surface also exposes
those values as provider metadata.

The OpenAI-compatible metadata extractor therefore now maps both snake_case provider payloads and
camelCase replay payloads into:

- `providerMetadata.deepseek.promptCacheHitTokens`
- `providerMetadata.deepseek.promptCacheMissTokens`

The typed DeepSeek metadata helper accepts both spellings while serializing the AI SDK-style
camelCase keys.

## Validation

Locked by:

- `cargo test -p siumai-core perplexity_metadata_helper_extracts_hosted_search_fields -- --nocapture`
- `cargo nextest run -p siumai-core deepseek_metadata_helper_extracts_prompt_cache_usage_fields --no-fail-fast`
- `cargo nextest run -p siumai-core deepseek_streaming_emits_usage_then_single_stream_end_and_ignores_done --no-fail-fast`
- `cargo nextest run -p siumai-provider-deepseek deepseek_metadata --no-fail-fast`
- `cargo nextest run -p siumai --test provider_public_path_parity_test --features google-vertex --no-fail-fast`
- `cargo nextest run -p siumai --test gemini_generate_content_stream_bridge_roundtrip_fixtures_alignment_test vertex_generate_content_stream_bridge_roundtrip_fixture_summary_cases_match --features google-vertex`
