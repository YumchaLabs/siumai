# Stream Metadata Parity Hardening - Milestones

Last updated: 2026-04-11

## Completed

- Reproduced the Perplexity public-path metadata mismatch caused by missing
  `providerMetadata.perplexity.usage.reasoningTokens`
- Reproduced the Vertex public-path reasoning custom-event mismatch caused by missing Gemini
  `reasoning-*` custom events
- Confirmed the shared stream-factory compatibility layer could duplicate textual shadow deltas
  when a converter already emitted the legacy lane directly
- Made textual shadow replay idempotent at the shared stream-factory boundary
- Restored Perplexity `reasoningTokens` into the typed hosted-search metadata surface
- Emitted Gemini / Vertex `reasoning-start` / `reasoning-delta` / `reasoning-end` custom events
  alongside stable runtime parts
- Suppressed duplicate reasoning-delta emission during Gemini GenerateContent bridge
  round-trips
- Revalidated all `provider_public_path_parity_test` cases under `--features google-vertex`

## Next

- Continue auditing other OpenAI-compatible typed metadata helpers for missing camelCase/usage
  fields compared with `repo-ref/ai`
- Keep checking mixed `Part + Custom` serializer paths for other providers as more stable runtime
  parts are enabled by default
