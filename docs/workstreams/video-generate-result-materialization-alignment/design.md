# Video Generate Result Materialization Alignment - Design

Last updated: 2026-04-21

## Context

The earlier `video-model-family-alignment` workstream already closed the biggest architectural
gaps:

- task-oriented `VideoModelV3` / `VideoModelV4` / `VideoModel`
- dedicated registry/factory video handles
- stable public `siumai::video` helpers
- high-level `generate(...)` batching and task polling

However, one high-value helper/result gap still remained against
`repo-ref/ai/packages/ai/src/generate-video/generate-video.ts` and
`repo-ref/ai/packages/ai/src/generate-video/generate-video-result.ts`:

- `generate(...)` still left URL-backed final videos unmaterialized by default
- `GenerateVideoResult` still lacked an AI SDK-style mandatory first `video`
- `GeneratedVideo` still lacked direct file-style `bytes()` / `base64()` accessors for the common
  already-materialized cases

That made the public Rust helper closer than before, but still not close enough at the result
boundary.

## Goal

- Make `siumai::video::generate(...)` materially closer to AI SDK `experimental_generateVideo()`
  on the audited URL/base64/bytes result paths.
- Align the result structs around the mandatory first `video` plus `videos`.
- Keep the honest Rust-first task model and explicit provider-reference limitations.

## Non-goals

- Do not fake AI SDK's `doGenerate(...)` on the core video family.
- Do not invent a generic provider-reference download contract.
- Do not remove `generate_materialized(...)`; it still has value as the explicit
  `MaterializedVideo` normalization helper.

## Chosen design

### 1. Materialize URL-backed final videos by default in `generate(...)`

`GenerateOptions` now controls helper-level final-video downloads directly:

- `materialize_urls: bool` defaults to `true`
- `materialize_http_config: Option<HttpConfig>` configures URL downloads when materialization is
  enabled

This keeps the task lifecycle explicit while making the high-level helper closer to the audited AI
SDK behavior.

The helper-level default now applies only to URL schemes that the Rust facade can truthfully
materialize on its own:

- `data:`
- `http:`
- `https:`

Non-downloadable provider-owned URL schemes such as current Vertex `gs://...` outputs stay
URL-backed and surface an explicit helper warning instead of failing the whole generate call.

### 2. Expose the AI SDK-style first `video` on result structs

Both result structs now expose a mandatory first item in addition to the full list:

- `GenerateVideoResult.video`
- `GenerateMaterializedVideoResult.video`

The compatibility helper methods remain, but the field now mirrors the upstream data shape more
honestly.

### 3. Make `GeneratedVideo` more file-like on the common path

`GeneratedVideo` now exposes:

- `url()` with metadata fallback so source URLs remain visible after helper-level materialization
- `bytes()` for inline byte/base64-backed videos
- `base64()` for inline byte/base64-backed videos

This narrows the everyday ergonomic gap against AI SDK `GeneratedFile` without pretending provider
references are generically downloadable.

### 4. Keep the real remaining gap explicit

The workstream intentionally does **not** erase the two real differences that still matter:

- the core Rust video family remains task-oriented rather than a fake `doGenerate(...)` callable
  model
- provider-reference-only assets still require provider-owned handling instead of a generic helper

## Validation

This slice is locked by:

- `siumai/src/video.rs` unit tests, including URL materialization default coverage
- `siumai/tests/public_surface_imports_test.rs`
- focused `cargo nextest run -p siumai --lib video`
- focused `cargo nextest run -p siumai --test public_surface_imports_test public_surface_video_family_imports_compile`

## Follow-up

- Provider-reference-only assets now have a dedicated follow-up workstream under
  `docs/workstreams/video-provider-reference-materialization-alignment/`; audited
  provider-owned download adapters are now wired for Gemini and MiniMaxi, while providers that
  still require a separate authenticated runtime (for example current Vertex GCS outputs) remain a
  narrower deferred follow-up.
