# ElevenLabs Audio Provider — Handoff

Status: Closed
Last updated: 2026-05-26

## Current State

The workstream is closed. ELA-010 froze the speech/transcription-only scope, ELA-020 added the
native ElevenLabs provider crate, ELA-030 wired ElevenLabs into registry metadata and family routing,
ELA-040 exposed the facade/public surface, and ELA-050 reran fresh focused gates and closed the lane.

## Active Task

- Task ID: none
- Owner: n/a
- Files:
  - `docs/workstreams/elevenlabs-audio-provider/*`
- Validation:
  - `cargo nextest run -p siumai-provider-elevenlabs --features elevenlabs --no-fail-fast`
  - `cargo nextest run -p siumai-registry --features elevenlabs elevenlabs --no-fail-fast`
  - `cargo nextest run -p siumai --features elevenlabs elevenlabs --no-fail-fast`
  - `cargo fmt --check -p siumai-provider-elevenlabs`
  - `cargo fmt --check -p siumai-registry`
  - `cargo fmt --check -p siumai`
  - `python -m json.tool docs\workstreams\elevenlabs-audio-provider\WORKSTREAM.json`
  - `git diff --check`
- Status: CLOSED
- Review: Closeout review found no blocking findings.
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

- Start separate follow-ons if needed: `elevenlabs-voice-resources` for voice listing/cloning and
  pronunciation-dictionary resources, `live-credential-smoke-tests` for optional provider smoke
  gates, or `media-task-polling-foundation` for queued media providers such as Fal or Replicate.
