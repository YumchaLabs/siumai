# Google Vertex Typed Option Surface Alignment - Todo

Last updated: 2026-04-20

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

## Open

- [ ] Decide whether the generic Rust-first `siumai::video::generate(...)` helper is sufficient
      for Vertex, or whether Vertex still needs extra provider-owned final-result/download helpers
      above the current task-based runtime
