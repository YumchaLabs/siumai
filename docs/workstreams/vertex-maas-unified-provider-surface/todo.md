# Vertex MaaS Unified Provider Surface - TODO

Last updated: 2026-04-10

Status legend:

- `[ ]` not started
- `[~]` in progress
- `[x]` done
- `[-]` intentionally deferred

## Track A - Canonical provider identity

- [x] Audit the AI SDK Vertex MaaS provider surface in `repo-ref/ai/packages/google-vertex/src/maas/*`.
- [x] Make `vertex-maas` a first-class built-in provider id.
- [x] Keep `google-vertex-maas` and `vertex.maas` as compatibility/discovery aliases.
- [x] Register native metadata and curated model ids for the provider catalog.

## Track B - Runtime/factory architecture

- [x] Reuse the shared OpenAI-compatible runtime for chat/completion/embedding.
- [x] Derive the OpenAPI base URL from `project + location`, env fallback, or explicit `base_url`.
- [x] Support Google-style Bearer auth through token providers and explicit `Authorization` headers.
- [x] Allow shared compat execution to proceed when auth is already present in headers.
- [-] Create a separate provider-owned `siumai-provider-google-vertex-maas` crate.
  - rejected for now because the AI SDK reference already treats MaaS as an OpenAI-compatible
    wrapper, and the registry-layer implementation is materially cheaper to maintain

## Track C - Public facade and regression coverage

- [x] Expose `Provider::vertex_maas()` on the public facade.
- [x] Expose `Siumai::builder().vertex_maas()` on the compatibility builder surface.
- [x] Add registry contract tests for base URL precedence, `global` fallback, auth injection, and
  completion/embedding family support.
- [x] Add public-path parity tests for builder/provider/registry equivalence.
- [x] Add public-surface compile guards for the unified builder semantics.
  - compile coverage now also pins `provider_ext::vertex_maas::{chat, completion, embedding,
    model_sets}` so the promoted public namespace stays aligned with the audited MaaS model subset
- [x] Add a negative regression test that locks image/rerank/speech/transcription registry handles
  as intentionally unsupported for `vertex-maas`.

## Track D - Stable typing and cross-layer support

- [x] Promote `ProviderType::VertexMaas` usage across provider catalog, retry defaults, and
  parameter validation.
- [x] Promote `ProviderType::Vertex` and `ProviderType::AnthropicVertex` so the broader Google
  Vertex wrapper family also stops degrading to `Custom(...)`.
- [x] Add catalog coverage that proves `vertex`, `anthropic-vertex`, and `vertex-maas` all resolve
  native metadata instead of generic custom-provider descriptions.

## Track E - Docs and follow-up

- [x] Record the design in `docs/workstreams/vertex-maas-unified-provider-surface/`.
- [x] Update the AI SDK structural-alignment workstream to mark Vertex MaaS as closed.
- [x] Update unreleased changelog sections instead of writing release notes.
- [x] Mirror package exports such as `GoogleVertexMaasProviderSettings` and `VERSION` on the Rust
  facade.
  - `GoogleVertexMaasProviderSettings` now supports the audited `project`, `location`, `baseURL`,
    `headers`, and `fetch` subset, plus a Rust token-provider analogue for the Node auth wrapper
    instead of modeling Node's `googleAuthOptions` object directly.
- [x] Revisit whether the broader `vertex` default-model story should become family-specific on the
  public facade instead of one provider-wide fallback.
  - `Siumai::builder().vertex()` and `Siumai::builder().anthropic_vertex()` now require an
    explicit `model`, matching the AI SDK family-specific `languageModel()` /
    `embeddingModel()` / `imageModel()` construction story instead of injecting a provider-wide
    default
- [ ] Revisit `vertex-maas` image-family support only if the AI SDK reference starts treating it as
  a documented and tested provider capability rather than a generic inherited interface method.
