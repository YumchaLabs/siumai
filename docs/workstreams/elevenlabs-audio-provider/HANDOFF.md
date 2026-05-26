# ElevenLabs Audio Provider — Handoff

Status: Active
Last updated: 2026-05-26

## Current State

The workstream is open, scope is frozen, and ELA-020 is complete. `siumai-provider-elevenlabs`
now provides the native speech/transcription provider crate following the closed Deepgram provider
pattern, with explicit differences for `xi-api-key` auth, voice-id TTS paths, and multipart STT
uploads.

## Active Task

- Task ID: ELA-030
- Owner: worker
- Files: `siumai-registry`, `siumai-core`, root `Cargo.toml`, `siumai-registry/Cargo.toml`
- Validation: `cargo nextest run -p siumai-registry --features elevenlabs elevenlabs --no-fail-fast`; `cargo fmt --check -p siumai-registry`
- Status: NEEDS_CONTEXT
- Review: review-workstream before accepting completion
- Evidence: `docs/workstreams/elevenlabs-audio-provider/EVIDENCE_AND_GATES.md`

## Decisions Since Last Update

- Use a dedicated provider crate: `siumai-provider-elevenlabs`.
- Keep scope to speech and transcription; no voice resource APIs, live credentials, Fal, Replicate, or queued media polling.
- Mirror AI SDK package names at facade/provider roots where they fit Rust naming, but do not widen `prelude::unified`.
- Treat the AI SDK lowercase `elevenlabs` alias as a facade decision in ELA-040, not a provider-crate blocker.
- Keep multipart materialization provider-local in ELA-020; no shared core transport API change was needed.

## Blockers

- None for ELA-030.

## Next Recommended Action

- Start ELA-030 by wiring ElevenLabs into registry feature flags, provider ids, native metadata,
  provider catalog, factory selection, and speech/transcription handle construction. Mirror Deepgram's
  registry factory shape, but preserve ElevenLabs-specific default models and unsupported-family messages.
