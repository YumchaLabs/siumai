# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Curated `google_vertex::{chat, embedding, image}` and `anthropic_vertex::chat` model constants
  are now exported from the provider package, so the public facade and registry catalog can reuse
  one audited curated subset instead of duplicating handwritten model lists.
- Native Google Vertex now also exposes the safe AI SDK-style typed option alias subset on the
  provider-owned/public surface:
  `GoogleVertexEmbeddingModelOptions`, `GoogleVertexImageModelOptions`, and deprecated
  `GoogleVertexImageProviderOptions`, all mapped onto the existing native embedding/Imagen typed
  option structures.
- Anthropic-on-Vertex now also exposes a dedicated AI SDK-style package settings wrapper
  `GoogleVertexAnthropicProviderSettings`, the audited Vertex-supported Anthropic tool subset under
  `providers::anthropic_vertex::{tools,provider_tools,hosted_tools}`, and the narrower Anthropic
  message-metadata names (`AnthropicMessageMetadata`, `AnthropicMessageContainerMetadata`,
  `AnthropicMessageContainerSkill`, `AnthropicUsageIteration`) on the wrapper surface. The same
  wrapper path now also exposes `GoogleVertexAnthropicMessagesModelId`, and its curated
  `models::{chat,ALL_CHAT}` subset is aligned with the current audited upstream model-id union.

### Fixed

- Vertex embedding response transformation now preserves an AI SDK-style response envelope with
  the raw response body on direct transformer usage.
- Vertex express-mode authentication now wins consistently when `GOOGLE_VERTEX_API_KEY` supplies
  the API key, suppressing token-provider auth just like the audited AI SDK node wrapper does when
  an effective API key is present.
- Anthropic-on-Vertex structured-output and reasoning streams now preserve the expected public
  client semantics again: indexed Anthropic textual/thinking blocks replay compatible
  `ContentDelta` / `ThinkingDelta` shadows on the public stream, and metadata-only redacted
  thinking placeholders no longer appear as empty reasoning strings on final responses.
- Vertex content-part metadata helpers now cover stable `reasoning-file` and `custom` parts
  alongside the older multimodal variants.
- Vertex Imagen request shaping now consumes canonical top-level `aspectRatio` / `seed` on both
  generation and edit paths, instead of requiring those controls to flow through provider-owned
  option maps or extra params.
- Vertex Imagen variation now also executes on the shared image-variation surface: the client
  advertises native variation support, variation requests transform into the Imagen `:predict`
  body with reference images plus optional prompt/negative-prompt controls, and URL-backed inputs
  work once the shared executor materializes them before the synchronous transformer runs.
- Anthropic-on-Vertex construction now mirrors the audited AI SDK package more honestly across the
  builder/settings/registry paths: explicit `base_url` remains supported, but provider builders and
  registry-backed unified construction can now also derive the canonical
  `/publishers/anthropic/models` base URL from explicit `project + location` or
  `GOOGLE_VERTEX_PROJECT` + `GOOGLE_VERTEX_LOCATION`.
- Anthropic-on-Vertex structured outputs now default to the same wrapper semantics as the audited
  AI SDK package: JSON-schema requests use the reserved `json` tool fallback by default on the
  Vertex path, and the streaming converter now receives the same effective structured-output mode
  so request shaping and streamed/final JSON extraction do not drift apart across model families.

## [0.11.0-beta.5] - 2026-01-15

### Added

- Dedicated Google Vertex AI provider crate (split from the previous Gemini/Google layer).
- Builder aliases and vercel-aligned tool exposure for Vertex setups.
- ADC auth auto-enable behavior (Vercel parity) and embedding batch size guards.

### Fixed

- Imagen API-key mode now appends `?key=...` to endpoint URLs.
- Anthropic-on-Vertex request shaping aligned with Vercel behavior.
