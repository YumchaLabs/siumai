# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Native OpenAI provider now also exposes a provider-owned `skills()` resource aligned with the
  AI SDK `OpenAISkills` surface: multipart/base64 helpers upload to `POST /v1/skills`, canonical
  `providerReference` / `providerMetadata` are returned from the stable result shape, and
  `defaultVersion` / `latestVersion` plus timestamps are preserved under the `openai` provider
  root.
- Native OpenAI provider now also exposes the main AI SDK-style typed option names on the
  provider-owned/public surface:
  `OpenAILanguageModel{Chat,Responses,Completion}Options`,
  `OpenAIEmbeddingModelOptions`, `OpenAISpeechModelOptions`,
  `OpenAITranscriptionModelOptions`, `OpenAIFilesOptions`, plus the deprecated upstream
  compatibility aliases `OpenAIChatLanguageModelOptions` and `OpenAIResponsesProviderOptions`.

### Fixed

- Native OpenAI completion now follows the AI SDK completion-family execution path directly:
  non-stream and streamed calls use the real `/completions` route, structured prompts are
  materialized with the audited completion rules, completion provider options normalize
  `logitBias` / `logprobs` / `user` on the OpenAI namespace, and completion responses preserve raw
  `choices[0].logprobs` provider metadata.
- Native OpenAI completion streaming now also honors runtime-only `includeRawChunks`: the audited
  `/completions` SSE path emits stable `stream-start`, `raw`, `response-metadata`, `text-*`, and
  terminal `finish` parts while preserving legacy `ContentDelta` / `StreamEnd`.
- Native OpenAI completion streaming now also preserves first-chunk parse-failure lifecycle
  ordering: invalid SSE payloads emit `stream-start` before optional runtime `raw` and the parse
  error, even when `includeRawChunks` is disabled.
- Native OpenAI completion streaming terminal responses now preserve raw provider finish reasons on
  stable `ChatResponse.raw_finish_reason` instead of dropping them at stream end.
- Responses input warning parity middleware now tolerates the expanded reasoning-part stable shape
  after the request-side `providerOptions` rollout.
- Responses input warning parity middleware now reads reasoning request state from canonical
  `providerOptions.openai`, preserves AI SDK-style warning snapshots for malformed reasoning
  provider options, and stops treating `provider_metadata` as a request-side reasoning carrier.
- OpenAI `skills()` upload now mirrors the audited AI SDK warning behavior for `displayTitle` by
  returning stable `unsupported { feature: "displayTitle" }` warnings instead of silently
  ignoring that option.
- Best-effort remote cancel for streaming Responses now tracks structured
  `Part(ResponseMetadata)` events in addition to the legacy `openai:response-metadata` custom
  event, so HTTP and websocket wrappers still call `POST /responses/{id}/cancel` after the
  stream-part migration.
- OpenAI transcription and audio-translation clients now consume the canonical shared `audio`
  request input directly, and no longer read `file_path` inside provider client code for
  translation or streaming transcription paths.
- OpenAI provider-side transcription and audio-translation request paths now also treat the shared
  transcription `mediaType` as required input, so multipart uploads always attach the canonical
  MIME type instead of silently omitting it.
- Completion SSE test coverage compiles again after restoring the missing `SseEventConverter`
  trait import on the audited streaming lane.
- OpenAI speech/transcription typed provider options now map onto real request behavior more
  closely: TTS provider options accept `speed`, transcription provider options accept
  `language` / `timestampGranularities`, and OpenAI audio request shaping now accepts both
  camelCase and snake_case option keys instead of depending on one internal JSON spelling.

## [0.11.0-beta.5] - 2026-01-15

### Added

- Native OpenAI provider extracted into its own crate as part of the workspace split.
- Vercel-aligned Responses API request/response shaping and fixtures.

### Fixed

- Preserve transcription text on SSE EOF.
- Default moderation model selection when no model is provided.
- Parsing/validation improvements for Files API and Responses stream events.
