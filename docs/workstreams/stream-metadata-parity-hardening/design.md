# Stream Metadata Parity Hardening - Design

Last updated: 2026-05-17

## Problem

The ongoing AI SDK alignment work exposed a second cluster of parity gaps around stream semantics
and provider metadata extraction:

- the shared stream factory previously replayed textual shadow deltas for stable runtime
  `TextDelta` / `ReasoningDelta` parts, but that replay could double-emit text/reasoning when a
  converter already emitted typed parts itself
- the Perplexity OpenAI-compatible metadata helper normalized hosted-search metadata into
  `providerMetadata.perplexity`, but it still dropped `usage.reasoningTokens`
- Gemini / Vertex reasoning streams emitted stable runtime parts, but they did not also expose
  AI SDK-style custom `reasoning-*` events on the public stream surface
- after adding those Gemini custom reasoning events, the Gemini GenerateContent serializer could
  see both the typed `Part` lane and the mirrored `Custom` lane during bridge round-trips and
  serialize the same reasoning delta twice
- the shared OpenAI-compatible metadata helper normalized DeepSeek prompt-cache accounting into
  stable `Usage`, but did not mirror AI SDK's `providerMetadata.deepseek.promptCacheHitTokens` /
  `promptCacheMissTokens` response metadata fields

These are all parity regressions of the same general kind: the stable runtime lane is present, but
the bridge/export metadata surface is still incomplete.

## Design

### 1. Keep textual streams typed-only

The shared stream factory keeps the canonical runtime rule:

- stable textual parts remain the primary runtime representation
- textual consumers read `ChatStreamPart::TextDelta` / `ReasoningDelta` directly

The earlier transitional shadow-replay layer has been removed with the typed-stream-only model.

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

### 3. Expose Gemini reasoning on typed parts

Gemini / Vertex reasoning chunks now emit:

- stable runtime `Part(ChatStreamPart::{ReasoningStart,ReasoningDelta,ReasoningEnd})`
- AI SDK-style custom `reasoning-start` / `reasoning-delta` / `reasoning-end`

The public stream surface therefore matches the practical AI SDK expectation:

- part-aware consumers can use the semantic lane
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

### 6. Promote standard OpenAI-compatible metadata extraction out of the provider allowlist

The config-driven OpenAI-compatible adapter now treats standard Chat Completions response metadata
as a family-level capability instead of a hard-coded provider allowlist. This matches the AI SDK
`createOpenAICompatible` behavior for generic providers:

- `sources`
- `choices[0].logprobs.content`
- `usage.completion_tokens_details.accepted_prediction_tokens`
- `usage.completion_tokens_details.rejected_prediction_tokens`

are extracted for every `ConfigurableAdapter` provider when present. Perplexity keeps its dedicated
hosted-search metadata extractor because its response shape carries additional provider-owned
fields.

Alias providers use the canonical provider metadata namespace before stream/non-stream transformers
apply any requested public key. This prevents legacy aliases such as `moonshot` from leaking
parallel `providerMetadata.moonshot` roots when the public API expects `providerMetadata.moonshotai`.

## Validation

Locked by:

- `cargo test -p siumai-core perplexity_metadata_helper_extracts_hosted_search_fields -- --nocapture`
- `cargo nextest run -p siumai-core deepseek_metadata_helper_extracts_prompt_cache_usage_fields --no-fail-fast`
- `cargo nextest run -p siumai-core deepseek_streaming_emits_usage_then_single_stream_end_and_ignores_done --no-fail-fast`
- `cargo nextest run -p siumai-provider-deepseek deepseek_metadata --no-fail-fast`
- `cargo nextest run -p siumai --test provider_public_path_parity_test --features google-vertex --no-fail-fast`
- `cargo nextest run -p siumai --test gemini_generate_content_stream_bridge_roundtrip_fixtures_alignment_test vertex_generate_content_stream_bridge_roundtrip_fixture_summary_cases_match --features google-vertex`
- `cargo nextest run -p siumai-protocol-openai --all-features --no-fail-fast`
- `cargo nextest run -p siumai-provider-openai-compatible --all-features --no-fail-fast`
