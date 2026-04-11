# Anthropic Vertex Stream Compatibility Alignment - Todo

Last updated: 2026-04-11

## Done

- [x] Reproduce the `anthropic_vertex` structured-output stream regression
- [x] Reproduce the `anthropic_vertex` reasoning stream regression
- [x] Reproduce the metadata-only empty reasoning accessor regression
- [x] Restore textual shadow deltas on SSE/JSON stream factory paths
- [x] Restore the same compatibility on transport-backed JSON streaming
- [x] Filter empty reasoning strings from response/message helper accessors
- [x] Add focused regression tests for the shared stream factory and reasoning helpers

## Open

- [ ] Audit whether other providers now rely on stable textual parts without compatible legacy
      shadows
- [ ] Decide whether compatibility shadow replay should eventually become configurable per stream
      consumer
