# Google Vertex Typed Option Surface Alignment - Todo

Last updated: 2026-04-11

## Done

- [x] Compare native `google_vertex` typed options with `repo-ref/ai/packages/google-vertex/src/index.ts`
- [x] Add `GoogleVertexEmbeddingModelOptions`
- [x] Add `GoogleVertexImageModelOptions`
- [x] Add deprecated `GoogleVertexImageProviderOptions`
- [x] Re-export the aliases through `provider_ext::google_vertex`
- [x] Add compile guards

## Open

- [ ] Revisit Vertex video only after native provider-owned video support exists
- [ ] Audit whether any future Vertex video work should live in this crate or a separate focused
      provider-owned module
