# DeepSeek Package Surface Alignment - TODO

Last updated: 2026-05-18

Status legend:

- `[ ]` not started
- `[~]` in progress
- `[x]` done
- `[-]` intentionally deferred

## Package Surface Audit

- [x] Audit `repo-ref/ai/packages/deepseek/src/index.ts` and
  `repo-ref/ai/packages/deepseek/src/deepseek-provider.ts`.
- [x] Confirm the upstream provider package is chat/language-model only.
- [x] Confirm upstream embedding and image model creation are intentionally unsupported.

## Rust Public Surface

- [x] Expose `provider_ext::deepseek::deepseek()` and `create_deepseek()`.
- [x] Expose `DeepSeekProviderSettings`, `DeepSeekErrorData`, and `VERSION`.
- [x] Expose AI SDK-style option aliases for the DeepSeek language/chat option surface.
- [x] Expose typed DeepSeek response metadata helpers.
- [x] Keep curated DeepSeek model constants scoped to chat models.

## Registry And Capability Boundary

- [x] Route text-family construction through the provider-owned `DeepSeekBuilder`.
- [x] Expose native registry language-model handles for `deepseek:<model>`.
- [x] Reject provider-owned embedding, image, speech, transcription, and rerank family paths.
- [x] Prevent promoted OpenAI-compatible completion inheritance from widening the DeepSeek package
  boundary.

## Evidence

- [x] Lock registry factory/catalog behavior with focused `siumai-registry` tests.
- [x] Lock facade package exports with `public_surface_deepseek_provider_ext_compiles`.
- [x] Lock public path request/stream behavior with `deepseek_public_path` parity tests.

## Follow-up

- [ ] Re-audit if upstream `@ai-sdk/deepseek` adds new public model families or package exports.
