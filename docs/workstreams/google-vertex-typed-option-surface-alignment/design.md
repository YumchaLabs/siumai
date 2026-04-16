# Google Vertex Typed Option Surface Alignment - Design

Last updated: 2026-04-11

## Problem

Compared with `repo-ref/ai/packages/google-vertex/src/index.ts`, Siumai's public
`google_vertex` typed option surface still had a naming-layer drift:

- native Rust users could reach `VertexEmbeddingOptions` and `VertexImagenOptions`
- but AI SDK-style names such as `GoogleVertexEmbeddingModelOptions` and
  `GoogleVertexImageModelOptions` were missing

The same AI SDK package also exports:

- deprecated `GoogleVertexImageProviderOptions`
- `GoogleVertexVideoModelOptions`
- deprecated `GoogleVertexVideoProviderOptions`
- `GoogleVertexVideoModelId`

The important distinction is that Siumai does **not** currently have a provider-owned Vertex video
runtime or typed provider-option surface in the native `google_vertex` crate, so blindly adding
video aliases would have created fake parity.

## Design

### 1. Add only the safe alias subset

This pass adds the AI SDK-style names that map directly onto real provider-owned Rust types today:

- `GoogleVertexEmbeddingModelOptions = VertexEmbeddingOptions`
- `GoogleVertexImageModelOptions = VertexImagenOptions`
- deprecated `GoogleVertexImageProviderOptions = GoogleVertexImageModelOptions`

These are thin aliases because the existing Rust implementation types already match the intended
provider-owned request lane closely enough.

### 2. Re-export the same aliases on provider and facade boundaries

The same names are now reachable through:

- `siumai_provider_google_vertex::provider_options::vertex::*`
- `siumai_provider_google_vertex::providers::vertex::*`
- `siumai::provider_ext::google_vertex::options::*`
- `siumai::provider_ext::google_vertex::*`

This keeps surface comparison against `repo-ref/ai` straightforward.

### 3. Do not fabricate Vertex video aliases yet

This pass intentionally does **not** add:

- `GoogleVertexVideoModelOptions`
- `GoogleVertexVideoProviderOptions`
- `GoogleVertexVideoModelId`

Reason:

- the current native `google_vertex` crate does not expose provider-owned Vertex video execution
  or typed video provider options
- adding those names now would imply runtime support that does not exist

The correct next step, if desired, is a real Vertex video implementation first, then a typed
public surface on top of that implementation.

## Validation

Locked by:

- `siumai/tests/public_surface_imports_test.rs`
- `cargo check -p siumai-provider-google-vertex --no-default-features --features google-vertex`
- `cargo check -p siumai --features google-vertex`
- `cargo nextest run -p siumai --test public_surface_imports_test --features google-vertex`
