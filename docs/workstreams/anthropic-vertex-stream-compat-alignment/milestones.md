# Anthropic Vertex Stream Compatibility Alignment - Milestones

Last updated: 2026-04-11

## Completed

- Reproduced the public `anthropic_vertex` regressions on reasoning and structured-output streams
- Confirmed the root cause split:
  - stable textual parts no longer replayed legacy textual delta shadows on public streams
  - metadata-only reasoning placeholders leaked through helper accessors as empty strings
- Restored textual shadow delta replay on the shared stream-factory / JSON-stream paths
- Filtered empty reasoning strings in `ChatResponse::reasoning()` and `ChatMessage::reasoning()`
- Revalidated `google-vertex` provider tests and stable public import guards

## Next

- Keep using the same "stable part lane + compatibility shadow lane" rule when auditing other
  providers against `repo-ref/ai`
- Revisit whether any non-textual stable parts need the same shared compatibility treatment later
