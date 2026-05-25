# Deepgram Audio Provider — Handoff

Status: Draft
Last updated: 2026-05-26

## Current State

The workstream is open. DGA-010 froze the execution lane from the closed AI SDK provider market expansion
decision. DGA-020 added the native Deepgram provider crate and passed focused provider-crate gates.

## Active Task

- Task ID: DGA-030
- Owner: worker
- Files:
  - `siumai-registry/*`
  - `siumai-core/*` if new feature metadata hooks are required
  - `Cargo.toml`
- Validation:
  - `cargo nextest run -p siumai-registry --features deepgram deepgram --no-fail-fast`
  - `cargo fmt --check -p siumai-registry`
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

## Blockers

- None currently.

## Next Recommended Action

- Start DGA-030 by wiring Deepgram into feature flags, registry metadata, and speech/transcription
  factory paths without widening the unified prelude.
