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

### Fixed

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

## [0.11.0-beta.5] - 2026-01-15

### Added

- Dedicated Google Vertex AI provider crate (split from the previous Gemini/Google layer).
- Builder aliases and vercel-aligned tool exposure for Vertex setups.
- ADC auth auto-enable behavior (Vercel parity) and embedding batch size guards.

### Fixed

- Imagen API-key mode now appends `?key=...` to endpoint URLs.
- Anthropic-on-Vertex request shaping aligned with Vercel behavior.
