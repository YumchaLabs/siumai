# Perplexity Package Surface Alignment - Design

Last updated: 2026-04-13

## Problem

Compared with `repo-ref/ai/packages/perplexity`, Siumai had already closed most of the public
package-boundary gaps on the Perplexity wrapper path:

- canonical `perplexity` identity already resolved as a first-class provider type
- top-level builder/provider/config/registry entrypoints already stayed chat-only, matching the
  audited AI SDK package boundary
- curated `sonar*` model constants/defaults already followed the current audited subset
- typed response metadata already used the AI SDK-shaped
  `providerMetadata.perplexity.{images,usage,cost}` layout

The remaining notable drift was lower-level but still user-visible on the public typed option
surface:

- `PerplexityOptions` serialized directly to Perplexity wire snake_case
  (`search_mode`, `return_images`, `web_search_options`, ...)
- that made the Rust provider-owned typed surface look like a transport contract instead of a
  package-level API
- the shared compat request boundary had no explicit Perplexity normalization step, so public typed
  options and wire payloads were effectively the same layer

That differs from the rest of the aligned provider surfaces in Siumai, where public typed options
stay AI SDK-style and the compat layer is responsible for lowering them onto the provider wire
shape.

## Goals

- Keep Perplexity on the audited chat/language-model-only package boundary.
- Make `PerplexityOptions` use AI SDK-style camelCase on the public typed surface.
- Keep backward-compatible snake_case input aliases for existing Rust callers.
- Add explicit compat normalization from public camelCase options to Perplexity wire snake_case.
- Document the boundary in a dedicated workstream instead of leaving it implicit in the global
  audit.

## Non-goals

- Do not invent embedding/image/completion/rerank/speech/transcription support on the Perplexity
  wrapper boundary.
- Do not add fake TypeScript-only exports such as `PerplexityProviderSettings` or `VERSION` on the
  Rust facade.
- Do not create a native Perplexity runtime outside the shared OpenAI-compatible transport.

## Chosen design

### 1. Public typed options stay package-level, not wire-level

`PerplexityOptions` and `PerplexityWebSearchOptions` now serialize with camelCase field names on
the Rust public surface:

- `searchMode`
- `searchRecencyFilter`
- `returnRelatedQuestions`
- `returnImages`
- `disableSearch`
- `enableSearchClassifier`
- `searchDomainFilter`
- `searchLanguageFilter`
- `searchAfterDateFilter`
- `searchBeforeDateFilter`
- `lastUpdatedAfterFilter`
- `lastUpdatedBeforeFilter`
- `imageDomainFilter`
- `imageFormatFilter`
- `webSearchOptions.searchContextSize`
- `webSearchOptions.userLocation`

This keeps provider-owned typed options aligned with the broader AI SDK-style public-shape rule
used elsewhere in Siumai.

### 2. Snake_case remains accepted as a compatibility alias

Serde aliases still accept the older snake_case field names on input, so existing Rust callers do
not have to migrate immediately.

This keeps the change source-compatible for deserialization/migration-heavy callers while moving
new typed serialization onto the canonical package-level shape.

### 3. Explicit compat normalization lowers onto Perplexity wire fields

The shared OpenAI-compatible request boundary now has a Perplexity-specific normalization step that
rewrites known public camelCase fields onto the provider wire contract:

- `searchMode -> search_mode`
- `searchRecencyFilter -> search_recency_filter`
- `returnRelatedQuestions -> return_related_questions`
- `returnImages -> return_images`
- `disableSearch -> disable_search`
- `enableSearchClassifier -> enable_search_classifier`
- `searchDomainFilter -> search_domain_filter`
- `searchLanguageFilter -> search_language_filter`
- `searchAfterDateFilter -> search_after_date_filter`
- `searchBeforeDateFilter -> search_before_date_filter`
- `lastUpdatedAfterFilter -> last_updated_after_filter`
- `lastUpdatedBeforeFilter -> last_updated_before_filter`
- `imageDomainFilter -> image_domain_filter`
- `imageFormatFilter -> image_format_filter`
- `webSearchOptions.searchContextSize -> web_search_options.search_context_size`
- `webSearchOptions.userLocation -> web_search_options.user_location`

Unknown extra Perplexity parameters still pass through unchanged.

### 4. Public typed options take precedence over legacy raw aliases

When both camelCase and snake_case variants are present for the same known option, the public
typed camelCase form wins during normalization.

That matches the broader Siumai rule for audited compat provider option surfaces: public package
shape is canonical, wire aliases are compatibility input only.

### 5. Package boundary stays chat-only

The audited `@ai-sdk/perplexity` package still exposes only the language-model/chat lane.

Siumai therefore keeps the Perplexity wrapper intentionally limited to:

- top-level chat generation
- chat streaming
- provider-owned typed request options
- provider-owned typed response metadata

and continues to reject:

- `completion_model(...)`
- `embedding_model(...)`
- `image_model(...)`
- `reranking_model(...)`
- `speech_model(...)`
- `transcription_model(...)`

## Validation

This workstream is locked by:

- typed option serialization/alias tests in
  `siumai-provider-openai-compatible/src/provider_options/perplexity.rs`
- request-extension tests in
  `siumai-provider-openai-compatible/src/providers/openai_compatible/ext/request_options.rs`
- compat normalization tests in
  `siumai-protocol-openai/src/standards/openai/compat/spec.rs`
- provider-runtime/public-path request parity tests in
  `siumai-provider-openai-compatible/src/providers/openai_compatible/openai_client.rs` and
  `siumai/tests/provider_public_path_parity_test.rs`

## Remaining follow-up

- Re-audit this workstream if the upstream AI SDK Perplexity package grows beyond the current
  chat/language-model-only boundary.
- Keep TypeScript-only package exports intentionally deferred unless a broader Rust package-facade
  pattern emerges first.
