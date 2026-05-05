# Google Vertex Typed Option Surface Alignment - Todo

Last updated: 2026-04-28

## Done

- [x] Compare native `google_vertex` typed options with `repo-ref/ai/packages/google-vertex/src/index.ts`
- [x] Add `GoogleVertexEmbeddingModelOptions`
- [x] Add `GoogleVertexImageModelOptions`
- [x] Add deprecated `GoogleVertexImageProviderOptions`
- [x] Add `GoogleVertexReferenceImage`
- [x] Add `GoogleVertexVideoModelOptions`
- [x] Add deprecated `GoogleVertexVideoProviderOptions`
- [x] Add `GoogleVertexVideoModelId`
- [x] Add `VertexVideoRequestExt`
- [x] Implement provider-owned Vertex video task creation/status runtime
- [x] Re-export the aliases through `provider_ext::google_vertex`
- [x] Add compile/public-path/registry guards
- [x] Expose the dedicated family-model constructor path for Vertex video.
  - `registry.video_model("vertex:...")` now resolves through the stable task-oriented video
    family handle instead of relying only on `language_model(...).create_video_task(...)`
- [x] Decide the high-level Vertex video helper boundary.
  - the generic Rust-first `siumai::video::generate(...)` helper is sufficient for the audited
    polling semantics
  - `VideoModel::polling_options(...)` / `VideoGenerationCapability::polling_options(...)` let
    the Vertex provider consume `providerOptions.vertex.pollIntervalMs` and `pollTimeoutMs` in the
    shared helper loop, matching AI SDK `doGenerate()` polling controls without sending those
    runtime-only fields to `predictLongRunning`
  - provider-owned final-result/download helpers remain deferred only for authenticated
    GCS-owned materialization; the generic helper must not pretend it can download `gs://...`
    without provider credentials

## Open

- [ ] Add a provider-owned authenticated Vertex GCS materialization helper if a real credentials
      path is introduced; do not add a fake generic downloader for `gs://...` outputs.
