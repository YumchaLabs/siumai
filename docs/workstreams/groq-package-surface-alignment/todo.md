# Groq Package Surface Alignment - TODO

Last updated: 2026-04-15

Status legend:

- `[ ]` not started
- `[~]` in progress
- `[x]` done
- `[-]` intentionally deferred

## Track A - Typed option parity

- [x] Audit `repo-ref/ai/packages/groq/src/groq-chat-options.ts`.
- [x] Add missing `GroqServiceTier::Performance`.
- [x] Add missing `GroqReasoningEffort::{Low, Medium, High}`.
- [x] Update validation plus config/builder/spec/client/provider-option tests for the expanded enum
  sets.
- [x] Add typed Groq helpers for the remaining audited chat option fields:
  `parallelToolCalls`, `user`, `structuredOutputs`, and `strictJsonSchema`.
- [x] Move the public `GroqLanguageModelOptions` serialization shape back to AI SDK-style
  camelCase provider options instead of leaking Groq wire-shape keys.
- [x] Keep config-first defaults and request-level typed options aligned by normalizing those
  provider options back to Groq wire keys before transport.
- [x] Ensure `GroqChatRequestExt::with_groq_options(...)` merges sibling Groq provider options
  instead of replacing the whole provider object.
- [x] Keep the AI SDK-aligned Groq `options::*` lane centered on
  `GroqLanguageModelOptions` / `GroqTranscriptionModelOptions`, while moving concrete
  `GroqSttOptions` / `GroqTtsOptions` discovery back under `ext::audio_options`.

## Track B - Model catalog parity

- [x] Audit Groq model-id drift against `repo-ref/ai/packages/groq/src/groq-chat-options.ts` and
  `repo-ref/ai/packages/groq/CHANGELOG.md`.
- [x] Refresh `KIMI_K2_INSTRUCT` to `moonshotai/kimi-k2-instruct-0905`.
- [x] Restore the missing current AI SDK chat model ids:
  `gemma2-9b-it`, `llama-guard-3-8b`, `llama3-{8b,70b}-8192`, `qwen-qwq-32b`,
  `qwen-2.5-32b`, and `deepseek-r1-distill-qwen-32b`.
- [x] Remove obsolete Groq catalog entries that are no longer part of the audited package
  boundary, including old system/vision/tool-use preview groupings.
- [x] Split provider-owned Groq TTS models out of the AI SDK-aligned chat catalog so chat and
  transcription parity can be audited without mixing in non-package extensions.

## Track C - Runtime/metadata parity

- [x] Compare `get-response-metadata.ts` and `groq-chat-language-model.ts` against the current Rust
  response-metadata lane.
- [x] Compare request shaping around reasoning / structured outputs / warnings against the current
  provider-owned Groq runtime.
- [x] Align the stable `groq-provider.ts` settings surface where the upstream package defines one:
  canonical `GROQ_API_KEY` env loading, AI SDK-style `headers`, and provider-construction parity
  across config-first and builder-first Rust entrypoints.
- [x] Keep the stable Groq facade's construction lane visible by re-exporting `GroqBuilder` under
  `provider_ext::groq::*` alongside the audited config/client surface.

## Track D - Docs and changelog

- [x] Create a dedicated `docs/workstreams/groq-package-surface-alignment/` folder.
- [x] Update `Unreleased` changelog entries for the typed enum, response metadata, and model
  catalog alignment.
- [x] Record `groq-provider.ts` settings-parity decisions in the workstream docs and `Unreleased`
  changelog sections.
- [ ] Fold future Groq package-surface parity fixes into this workstream instead of the
  browser-search-specific workstream.
