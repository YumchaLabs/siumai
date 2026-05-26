# ElevenLabs Audio Provider — Handoff

Status: Active
Last updated: 2026-05-26

## Current State

The workstream is open, scope is frozen, and ELA-020 through ELA-040 are complete.
`siumai-provider-elevenlabs` provides the native speech/transcription provider crate,
`siumai-registry` wires ElevenLabs into provider metadata and family routing, and the `siumai`
facade now exposes `provider_ext::elevenlabs`, `providers::elevenlabs`, `Provider::elevenlabs()`,
model constants, typed options, request extension traits, and build-time provider accounting without
widening `prelude::unified`.

## Active Task

- Task ID: ELA-050
- Owner: planner
- Files: `docs/workstreams/elevenlabs-audio-provider`
- Validation: closeout gate set in `EVIDENCE_AND_GATES.md`
- Status: READY
- Review: review-workstream for final lane acceptance
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
- Mirror the AI SDK lowercase package export through `provider_ext::elevenlabs::elevenlabs()` and
  `siumai::providers::elevenlabs::elevenlabs()`, with `create_elevenlabs()` as the Rust analogue of
  `createElevenLabs()`.
- Keep ElevenLabs-specific options and extension traits scoped under `provider_ext::elevenlabs`;
  do not add them to `prelude::unified`.

## Blockers

- None for ELA-040.

## Next Recommended Action

- Start ELA-050 closeout: run final verification, review the task ledger and evidence, then either
  close the lane or split any residual voice/resource/live-credential gaps into follow-ons.
