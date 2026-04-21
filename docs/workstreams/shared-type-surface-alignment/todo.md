# Shared Type Surface Alignment - TODO

Last updated: 2026-04-21

Status legend:

- `[ ]` not started
- `[~]` in progress
- `[x]` done
- `[-]` intentionally deferred

## Track A - Shared public names

- [x] Audit `repo-ref/ai/packages/ai/src/types/index.ts` and the upstream shared metadata/usage
  files it re-exports.
- [x] Add a dedicated `siumai-spec/src/types/ai_sdk.rs` module for shared AI SDK-style public
  names.
- [x] Expose `JSONValue`, `CallWarning`, `ProviderMetadata`, and `ImageModelProviderMetadata`.
- [x] Expose `ProviderOptions` and `Context`.
- [x] Expose provider-utils-style `ToolCall` and `ToolResult`.
- [x] Expose `EmbeddingModelUsage`, `ImageModelUsage`, and `LanguageModelUsage`.
- [x] Expose `LanguageModelRequestMetadata`, `LanguageModelResponseMetadata`,
  `ImageModelResponseMetadata`, `SpeechModelResponseMetadata`, and
  `TranscriptionModelResponseMetadata`.

## Track B - Runtime carrier parity

- [x] Add optional `headers` to stable `ResponseMetadata`.
- [x] Add the shared warning `deprecated` category to `Warning`.
- [x] Add honest conversions from `Usage`, `HttpRequestInfo`, `HttpResponseInfo`, and
  `ResponseMetadata` into the new shared public structs.
- [x] Keep speech response `body` optional instead of fabricating runtime capture.

## Track C - Facade and tests

- [x] Re-export the new shared names from `siumai::prelude::unified::*`.
- [x] Add public compile-guard coverage in `siumai/tests/public_surface_imports_test.rs`.
- [x] Add local unit coverage for warning/metadata/usage conversion behavior.

## Track D - Docs and changelog

- [x] Create a dedicated `docs/workstreams/shared-type-surface-alignment/` folder.
- [x] Record the shared type-surface alignment slice in `CHANGELOG.md` `Unreleased`.

## Track E - Intentional deferrals

- [-] Do not expose `RequestOptions`, `TimeoutConfiguration`, or `LanguageModelCallOptions` in
  this workstream without a separate design.
- [-] Do not pretend that every provider already captures speech/transcription response bodies in a
  stable cross-provider way.
