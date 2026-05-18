# Stream Metadata Parity Hardening - Todo

Last updated: 2026-05-17

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
- [x] Audit DeepSeek OpenAI-compatible provider metadata and restore AI SDK-style prompt cache
      hit/miss token fields
- [x] Promote standard OpenAI-compatible metadata extraction from a provider allowlist to all
      config-driven providers, including generic/custom providers
- [x] Preserve canonical provider metadata aliases such as `moonshot` -> `moonshotai` while still
      honoring requested public metadata keys like `testProvider`
- [x] Re-audit current Gemini/Anthropic/OpenAI stream serializers for mixed `Part + Custom`
      duplication; no new duplicate path was found beyond the existing typed-only and tool
      dedupe gates

## Future Triggers

- [-] Audit other bridge serializers for mixed-lane `Part + Custom` duplicate effects as more
      stable runtime parts become first-class
- [-] Revisit provider-specific metadata helpers when upstream AI SDK packages add response
      metadata fields beyond the shared OpenAI-compatible subset
