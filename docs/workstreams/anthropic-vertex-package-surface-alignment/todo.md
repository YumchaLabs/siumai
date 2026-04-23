# Anthropic Vertex Package Surface Alignment - TODO

Last updated: 2026-04-22

Status legend:

- `[ ]` not started
- `[~]` in progress
- `[x]` done
- `[-]` intentionally deferred

## Track A - Package export audit

- [x] Audit `repo-ref/ai/packages/google-vertex/src/anthropic/index.ts`.
- [x] Audit `repo-ref/ai/packages/google-vertex/src/anthropic/google-vertex-anthropic-provider.ts`.

## Track B - Constructor and settings parity

- [x] Add `GoogleVertexAnthropicProviderSettings` on the provider-owned/public Rust surface.
- [x] Add `vertex_anthropic()` / `create_vertex_anthropic()` on the stable public facade.
- [x] Add `Provider::vertex_anthropic()` alias for package-name comparability.
- [x] Add `Siumai::builder().vertex_anthropic()` alias on the unified-builder path.

## Track C - Base URL derivation parity

- [x] Add a shared `google_vertex_anthropic_base_url(...)` helper.
- [x] Let `VertexAnthropicBuilder` derive base URL from `project + location`.
- [x] Let `GoogleVertexAnthropicProviderSettings` forward `project + location` without forcing
  explicit `base_url`.
- [x] Let the registry/unified-builder `anthropic-vertex` factory derive the same base URL from
  explicit fields or `GOOGLE_VERTEX_PROJECT` + `GOOGLE_VERTEX_LOCATION`.
- [x] Keep explicit `base_url` as the highest-priority override.

## Track D - Tools and metadata surface

- [x] Expose the audited Vertex-supported Anthropic tool subset under
  `providers::anthropic_vertex::tools`.
- [x] Mirror `provider_tools` and `hosted_tools` on the public facade.
- [x] Re-export `AnthropicMessageMetadata` and the narrowed container/iteration structs on the
  Anthropic-on-Vertex facade.

## Track E - Model-id parity

- [x] Re-audit `google-vertex-anthropic-messages-options.ts`.
- [x] Add `GoogleVertexAnthropicMessagesModelId` on the provider-owned/public Rust surface.
- [x] Replace the stale curated Anthropic-on-Vertex constants with the current audited upstream
  model-id subset.
- [x] Lock the curated subset through provider and public-surface tests.

## Track F - Validation and docs

- [x] Lock builder/settings derivation behavior with provider crate tests.
- [x] Extend `siumai/tests/public_surface_imports_test.rs` for the new settings/constructors/tools.
- [x] Extend `siumai/tests/anthropic_vertex_builder_alignment_test.rs` for project/location paths.
- [x] Lock Vertex Anthropic structured-output default behavior to the AI SDK wrapper contract:
  default JSON-schema requests use the reserved `json` tool fallback and stream conversion receives
  the same effective mode.
- [x] Add a dedicated workstream folder under `docs/workstreams/`.
- [x] Update `docs/workstreams/ai-sdk-structural-alignment/data-structure-matrix.md`.
- [x] Update `CHANGELOG.md` and `siumai-provider-google-vertex/CHANGELOG.md`.

## Track G - Intentional deferrals

- [-] Do not fabricate a callable TypeScript-style provider object on the Rust facade.
- [-] Do not widen Anthropic-on-Vertex into unsupported non-text model families just to mirror
  generic provider traits.
