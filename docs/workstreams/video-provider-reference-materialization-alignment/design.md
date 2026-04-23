# Video Provider Reference Materialization Alignment - Design

Last updated: 2026-04-21

## Context

The earlier video workstreams already closed most of the structural gap against
`repo-ref/ai/packages/ai/src/generate-video/generate-video.ts`:

- stable task-oriented `VideoModelV3` / `VideoModelV4` / `VideoModel`
- dedicated registry `video_model(...)` handles
- high-level `siumai::video::generate(...)`
- default URL materialization on the helper path
- AI SDK-style first `video` plus `videos`

That left one real runtime gap on the Rust side:

- some providers return final video assets only as provider-owned references instead of bytes or
  direct URLs
- `generate(...)` could only auto-materialize URL-backed assets
- `GeneratedVideo::materialize(...)` cannot solve this generically because a standalone
  `GeneratedVideo` no longer owns the provider client/model needed to fetch provider-managed files

This is not an AI SDK shared-type gap. AI SDK `GeneratedFile` still assumes the helper already
returned bytes/base64. The remaining Rust gap was the provider-owned fetch bridge needed to reach
that same end state on audited providers.

## Goal

- Add a small shared contract that lets task-oriented video models materialize provider-owned final
  video references into bytes.
- Use that contract from the high-level `siumai::video::generate(...)` helper without hardcoding
  provider branches into the facade layer.
- Land audited implementations for the current providers that already have the needed provider-owned
  file-download runtime (`Gemini` and `MiniMaxi`).

## Non-goals

- Do not invent a generic provider-agnostic post-hoc download API on `GeneratedVideo`.
- Do not pretend all provider references are portable URLs.
- Do not force unsupported providers such as current Vertex GCS-owned video outputs into a fake
  shared download path before the auth/runtime story exists.

## Chosen design

### 1. Add a tiny shared materialization carrier

`siumai-spec/src/types/video.rs` now exposes `MaterializedVideoAsset`:

- `bytes: Vec<u8>`
- `media_type: Option<String>`

This keeps the core contract minimal and reusable. The higher-level `siumai::video` facade still
owns conversion into `GeneratedVideo` / `MaterializedVideo`.

### 2. Extend the existing video capability instead of forking a second trait hierarchy

`VideoGenerationCapability` and `VideoModelV3` now expose:

- `materialize_video_reference(&ProviderReference) -> MaterializedVideoAsset`

The default implementation stays explicitly unsupported.

That keeps the dispatch architecture simple:

- direct provider clients
- `Siumai`
- registry-backed `VideoModelHandle`
- `ClientBackedVideoModel`

all converge on the same video-family capability instead of growing a parallel provider-reference
trait.

### 3. Reuse audited provider-owned file runtimes

The first supported implementations are intentionally narrow:

- `Gemini`: resolves `providerReference["gemini" | "google"]` and reuses existing
  `FileManagementCapability::get_file_content(...)`
- `MiniMaxi`: resolves `providerReference["minimaxi"]` and reuses the same shared file-management
  path

No new provider-specific file transport was invented for this workstream. The design only wires the
already-audited provider-owned runtimes into the video-family helper path.

### 4. Keep `generate(...)` best-effort and honest

`siumai::video::generate(...)` now performs two materialization passes:

1. URL-backed final videos, controlled by `GenerateOptions.materialize_urls`
2. provider-reference-backed final videos, using the new model capability

For the URL-backed pass, the helper only auto-materializes schemes it can actually download on its
own (`data:`, `http:`, `https:`). Provider-owned URLs such as current Vertex `gs://...` outputs
stay URL-backed with a warning rather than being forced through a fake generic downloader.

Behavior is intentionally asymmetric:

- supported provider-owned references are materialized into byte-backed `GeneratedVideo`
- unsupported providers keep the original `ProviderReference`
- actual provider/runtime failures still surface as errors
- task-oriented status responses can now also carry canonical `providerReference` directly instead
  of forcing the facade to infer everything from legacy `fileId`

This keeps the helper closer to AI SDK `GeneratedFile` semantics where possible without pretending
that all provider-owned references are now generically downloadable.

## Validation

This slice is locked by:

- `siumai-core/src/video.rs` adapter regression coverage
- `siumai/src/video.rs` helper tests for supported and unsupported provider-reference materialization
- `siumai/tests/public_surface_imports_test.rs`
- `siumai/tests/provider_public_path_parity_test.rs` for Gemini canonical `providerReference`
  query responses and the intentionally unsupported Vertex raw-URL path
- focused `cargo nextest run -p siumai-core --lib adapter_materialize_video_reference_uses_capability`
- focused `cargo nextest run -p siumai --lib video`
- focused `cargo nextest run -p siumai --test public_surface_imports_test public_surface_video_family_imports_compile`
- targeted `cargo check -p siumai-provider-gemini -p siumai-provider-minimaxi -p siumai-core -p siumai-registry -p siumai`

## Follow-up

- Revisit provider-owned video outputs that require a different authenticated download runtime
  (for example current Vertex GCS-owned outputs) once a shared, audited path exists.
