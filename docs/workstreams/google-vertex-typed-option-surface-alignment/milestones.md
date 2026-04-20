# Google Vertex Typed Option Surface Alignment - Milestones

Last updated: 2026-04-20

## Completed

- Audited `repo-ref/ai/packages/google-vertex/src/index.ts`
- Kept embedding/image AI SDK aliases as thin names over the existing provider-owned Rust types
- Added a real provider-owned Vertex Veo task runtime (`:predictLongRunning` / `:fetchPredictOperation`)
- Added `GoogleVertexReferenceImage`
- Added `GoogleVertexVideoModelOptions`
- Added deprecated `GoogleVertexVideoProviderOptions`
- Added `GoogleVertexVideoModelId`
- Added `VertexVideoRequestExt`
- Re-exported the typed video surface from the provider-owned and public facade boundaries
- Added the dedicated stable registry/facade video lane (`registry.video_model("vertex:...")` +
  `siumai::video::{create_task, query_task}`) on top of the same provider-owned Veo runtime
- Added compile, public-path, and lower-contract coverage for the new video lane
