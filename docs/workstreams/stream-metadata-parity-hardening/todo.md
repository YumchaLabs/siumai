# Stream Metadata Parity Hardening - Todo

Last updated: 2026-04-11

## Done

- [x] Reproduce duplicate textual shadow deltas in the shared stream factory
- [x] Make textual shadow replay idempotent when legacy deltas already exist
- [x] Reproduce missing Perplexity `reasoningTokens` on typed provider metadata
- [x] Restore `providerMetadata.perplexity.usage.reasoningTokens`
- [x] Reproduce missing Vertex/Gemini public `reasoning-*` custom events
- [x] Emit Gemini reasoning start/delta/end on both stable part and AI SDK custom lanes
- [x] Reproduce Gemini GenerateContent bridge round-trip duplicate reasoning deltas
- [x] Suppress duplicate mixed-lane reasoning deltas in the Gemini serializer
- [x] Revalidate `provider_public_path_parity_test` under `google-vertex`

## Open

- [ ] Audit other OpenAI-compatible provider metadata helpers for missing typed usage fields
- [ ] Audit other bridge serializers for mixed-lane `Part + Custom` duplicate effects as more
      stable runtime parts become first-class
