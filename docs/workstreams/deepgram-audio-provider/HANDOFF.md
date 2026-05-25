# Deepgram Audio Provider — Handoff

Status: Draft
Last updated: 2026-05-26

## Current State

The workstream is open. DGA-010 froze the execution lane from the closed AI SDK provider market expansion
decision. DGA-020 added the native Deepgram provider crate and passed focused provider-crate gates.
DGA-030 wired Deepgram into the registry and passed focused registry gates. DGA-040 exposed the
Deepgram facade/public surface and passed focused facade gates.

## Active Task

- Task ID: DGA-050
- Owner: planner
- Files:
  - `docs/workstreams/deepgram-audio-provider/*`
- Validation:
  - verify-rust-workstream records fresh final gate evidence
  - `python -m json.tool docs\workstreams\deepgram-audio-provider\WORKSTREAM.json`
  - `git diff --check`
- Status: READY_TO_CLOSE
- Review: Pending closeout review
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
- DGA-040 added the facade `deepgram` feature/dependency, build-time provider accounting,
  `provider_ext::deepgram`, `providers::deepgram`, `Provider::deepgram()`, model catalog re-exports,
  and focused public-surface tests. The unified prelude remains unchanged.
- No optional example was added for DGA-040; the slice stayed focused on stable public paths and
  compile-time import coverage.

## Blockers

- None currently.

## Next Recommended Action

- Run DGA-050 closeout: review the shipped provider/registry/facade evidence, run final JSON and
  whitespace gates, then close the Deepgram lane or split narrow follow-ons.
