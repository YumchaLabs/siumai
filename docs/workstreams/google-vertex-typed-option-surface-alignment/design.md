# Google Vertex Typed Option Surface Alignment - Design

Last updated: 2026-04-20

## Problem

Compared with `repo-ref/ai/packages/google-vertex/src/index.ts`, Siumai's public
`google_vertex` typed option surface previously had a naming-layer drift:

- native Rust users could reach `VertexEmbeddingOptions` and `VertexImagenOptions`
- but AI SDK-style names such as `GoogleVertexEmbeddingModelOptions` and
  `GoogleVertexImageModelOptions` were missing

The same AI SDK package also exports:

- deprecated `GoogleVertexImageProviderOptions`
- `GoogleVertexVideoModelOptions`
- deprecated `GoogleVertexVideoProviderOptions`
- `GoogleVertexVideoModelId`

The earlier important distinction was that Siumai did **not** yet have a provider-owned Vertex
video runtime. That meant adding AI SDK video names first would have created fake parity.

That constraint is now gone: the native `google_vertex` crate owns a real Vertex video task path,
so the public typed video surface can be added honestly rather than as naming-only sugar.

## Design

### 1. Keep embedding/image aliases thin

The embedding/image names remain thin aliases over the existing provider-owned Rust types:

- `GoogleVertexEmbeddingModelOptions = VertexEmbeddingOptions`
- `GoogleVertexImageModelOptions = VertexImagenOptions`
- deprecated `GoogleVertexImageProviderOptions = GoogleVertexImageModelOptions`

These are thin aliases because the existing Rust implementation types already match the intended
provider-owned request lane closely enough.

### 2. Add provider-owned Vertex video typing on top of the real runtime

This pass now also exposes the typed video surface because the provider crate owns real Veo task
execution:

- `GoogleVertexReferenceImage`
- `GoogleVertexVideoModelOptions`
- deprecated `GoogleVertexVideoProviderOptions`
- `GoogleVertexVideoModelId`
- `VertexVideoRequestExt`

The runtime stays Rust-first:

- task creation uses `:predictLongRunning`
- task status uses `:fetchPredictOperation`
- the public Rust contract stays task-based (`create_video_task` / `query_video_task`)
- the stable family/registry lane now also exists through `registry.video_model("vertex:...")`
  plus `siumai::video::{create_task, query_task}`
- the older `language_model("vertex:...").create_video_task(...)` route remains only as a
  compatibility delegation path
- AI SDK-style polling option names are accepted on the typed option surface, but remain warning-only
  in Rust because polling is not hidden behind an auto-polling callable model object

### 3. Re-export the same names on provider and facade boundaries

The same names are now reachable through:

- `siumai_provider_google_vertex::provider_options::vertex::*`
- `siumai_provider_google_vertex::providers::vertex::*`
- `siumai::provider_ext::google_vertex::options::*`
- `siumai::provider_ext::google_vertex::*`

This keeps surface comparison against `repo-ref/ai` straightforward.

Video names now participate in that same provider-owned/public boundary, so surface comparison
against `repo-ref/ai` is straightforward without inventing TypeScript-only callable provider
exports.

## Validation

Locked by:

- `siumai/tests/public_surface_imports_test.rs`
- `siumai/tests/provider_public_path_parity_test.rs`
- `siumai-registry/src/registry/factories/contract_tests.rs`
- `cargo check -p siumai-provider-google-vertex --features google-vertex`
- `cargo nextest run -p siumai-provider-google-vertex --features google-vertex`
- `cargo nextest run -p siumai --test public_surface_imports_test --features google-vertex`
