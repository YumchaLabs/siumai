# Deepgram Audio Provider — Handoff

Status: Draft
Last updated: 2026-05-26

## Current State

The workstream is open. DGA-010 froze the execution lane from the closed AI SDK provider market expansion
decision. DGA-020 added the native Deepgram provider crate and passed focused provider-crate gates.
DGA-030 wired Deepgram into the registry and passed focused registry gates.

## Active Task

- Task ID: DGA-040
- Owner: worker
- Files:
  - `siumai/*`
  - `siumai/tests/*`
  - `examples/*`
- Validation:
  - `cargo nextest run -p siumai --features deepgram deepgram --no-fail-fast`
  - targeted public-surface import test
  - `cargo fmt --check -p siumai`
- Status: READY_TO_IMPLEMENT
- Review: Pending after implementation
- Evidence: `docs/workstreams/deepgram-audio-provider/EVIDENCE_AND_GATES.md`

## Decisions Since Last Update

- Deepgram is a native audio provider crate candidate, not an OpenAI-compatible preset.
- The first lane scope is speech and transcription only.
- Use `DEEPGRAM_API_KEY` fallback and `authorization: Token <key>` to match AI SDK.
- Expose provider-owned typed options and model constants under a stable Deepgram provider root.
- Do not widen `prelude::unified`.
- Do not include ElevenLabs, Replicate, Fal, or common media polling in this lane.
- DGA-020 intentionally stops before registry/facade wiring. The crate exposes `DeepgramClient`,
  `DeepgramSpeechModel`, `DeepgramTranscriptionModel`, `DeepgramConfig`, model constants, typed
  speech/transcription options, and request extension traits under `providers::deepgram`.
- Deepgram STT uses raw audio bytes with `Content-Type` from `SttRequest.media_type`; the custom
  no-network hook currently travels through the existing `HttpTransport::execute_multipart` byte
  request shape because there is no separate raw-body transport trait.
- DGA-030 added `siumai-registry` and `siumai-core` `deepgram` feature flags, native provider
  metadata, catalog model listing, `DeepgramProviderFactory`, `SiumaiBuilder::deepgram()`, and
  focused no-network registry tests. Deepgram's native metadata uses an explicit-model policy to
  avoid ambiguity between speech and transcription defaults on compatibility construction.
- The compatibility builder now routes known Deepgram transcription models (for example `nova-3`)
  through the transcription client before the generic audio speech-first fallback.

## Blockers

- None currently.

## Next Recommended Action

- Start DGA-040 by adding facade/provider extension exports and public import tests without widening
  `prelude::unified`.
