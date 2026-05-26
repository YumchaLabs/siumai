# ElevenLabs Audio Provider — Handoff

Status: Active
Last updated: 2026-05-26

## Current State

The workstream is open, scope is frozen, and ELA-020 plus ELA-030 are complete.
`siumai-provider-elevenlabs` provides the native speech/transcription provider crate, and
`siumai-registry` now wires ElevenLabs into feature flags, provider ids, native metadata, provider
catalog, factory selection, builder routing, and speech/transcription handle construction.

## Active Task

- Task ID: ELA-040
- Owner: worker
- Files: `siumai`, `siumai/tests`, `examples`
- Validation: `cargo nextest run -p siumai --features elevenlabs elevenlabs --no-fail-fast`; `cargo fmt --check -p siumai`
- Status: READY
- Review: review-workstream before accepting completion
- Evidence: `docs/workstreams/elevenlabs-audio-provider/EVIDENCE_AND_GATES.md`

## Decisions Since Last Update

- Use a dedicated provider crate: `siumai-provider-elevenlabs`.
- Keep scope to speech and transcription; no voice resource APIs, live credentials, Fal, Replicate, or queued media polling.
- Mirror AI SDK package names at facade/provider roots where they fit Rust naming, but do not widen `prelude::unified`.
- Treat the AI SDK lowercase `elevenlabs` alias as a facade decision in ELA-040, not a provider-crate blocker.
- Keep multipart materialization provider-local in ELA-020; no shared core transport API change was needed.
- Include `ProviderType::ElevenLabs` in `siumai-spec` for registry catalog compatibility metadata;
  this is the only intentional ELA-030 scope expansion beyond the original registry/core/Cargo files.
- Keep ElevenLabs out of OpenAI-compatible model normalization paths; it is a native audio provider.

## Blockers

- None for ELA-040.

## Next Recommended Action

- Start ELA-040 by exposing `provider_ext::elevenlabs`, `providers::elevenlabs`, facade feature
  wiring, model constants, typed options, and public import tests. Preserve the export policy:
  no widening of `prelude::unified`.
