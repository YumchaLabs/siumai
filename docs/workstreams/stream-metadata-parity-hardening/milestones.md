# Stream Metadata Parity Hardening - Milestones

Last updated: 2026-05-17

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
- Promoted standard OpenAI-compatible response metadata extraction from a provider allowlist to the
  config-driven adapter family, including generic/custom providers
- Locked alias-safe metadata namespace normalization for legacy `moonshot` -> `moonshotai`

## Future Triggers

- Keep checking mixed `Part + Custom` serializer paths for other providers as more stable runtime
  parts are enabled by default
- Revisit provider-specific metadata helpers only when new AI SDK provider packages add response
  metadata fields beyond the shared OpenAI-compatible subset
