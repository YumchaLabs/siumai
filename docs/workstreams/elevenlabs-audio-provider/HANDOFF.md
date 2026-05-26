# ElevenLabs Audio Provider — Handoff

Status: Active
Last updated: 2026-05-26

## Current State

The workstream is open and the scope is frozen. ElevenLabs is a native speech/transcription provider
lane following the closed Deepgram provider pattern, with explicit differences for `xi-api-key` auth,
voice-id TTS paths, and multipart STT uploads.

## Active Task

- Task ID: ELA-020
- Owner: worker
- Files: `siumai-provider-elevenlabs`, root `Cargo.toml`
- Validation: `cargo nextest run -p siumai-provider-elevenlabs --features elevenlabs --no-fail-fast`; `cargo fmt --check -p siumai-provider-elevenlabs`
- Status: NEEDS_CONTEXT
- Review: review-workstream before accepting completion
- Evidence: `docs/workstreams/elevenlabs-audio-provider/EVIDENCE_AND_GATES.md`

## Decisions Since Last Update

- Use a dedicated provider crate: `siumai-provider-elevenlabs`.
- Keep scope to speech and transcription; no voice resource APIs, live credentials, Fal, Replicate, or queued media polling.
- Mirror AI SDK package names at facade/provider roots where they fit Rust naming, but do not widen `prelude::unified`.
- Treat the AI SDK lowercase `elevenlabs` alias as a facade decision in ELA-040, not a provider-crate blocker.

## Blockers

- None for ELA-020.
- Multipart request testing may require either existing transport support or a narrow provider-local helper; record the chosen path in the ELA-020 handoff.

## Next Recommended Action

- Start ELA-020 by copying the proven Deepgram provider-crate shape and replacing the wire contract with ElevenLabs-specific auth, model constants, speech options, transcription options, TTS path behavior, and multipart STT tests.
