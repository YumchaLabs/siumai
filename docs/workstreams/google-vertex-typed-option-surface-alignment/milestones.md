# Google Vertex Typed Option Surface Alignment - Milestones

Last updated: 2026-04-11

## Completed

- Audited `repo-ref/ai/packages/google-vertex/src/index.ts`
- Confirmed native Siumai `google_vertex` only has provider-owned embedding/image typed options
- Added AI SDK-style alias names for the safe embedding/image subset
- Re-exported the aliases from the provider-owned and public facade boundaries
- Added compile guards on the stable facade

## Deferred on purpose

- `GoogleVertexVideoModelOptions`
- `GoogleVertexVideoProviderOptions`
- `GoogleVertexVideoModelId`

These remain deferred until the native provider crate gains real Vertex video runtime support.
