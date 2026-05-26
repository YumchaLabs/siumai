# ElevenLabs Audio Provider — TODO

Status: Active
Last updated: 2026-05-26

## ELA-010 — Scope And Contract Freeze

- [x] ELA-010 [owner=planner] [deps=none] [scope=docs/workstreams/elevenlabs-audio-provider,repo-ref/ai/packages/elevenlabs]
  Goal: Freeze the AI SDK package contract, Siumai implementation boundary, non-goals, task split, and validation gates.
  Validation: DESIGN.md, MILESTONES.md, EVIDENCE_AND_GATES.md, WORKSTREAM.json, and HANDOFF.md exist and agree.
  Review: planner self-review for task boundaries before implementation.
  Evidence: `docs/workstreams/elevenlabs-audio-provider/DESIGN.md`
  Handoff: DONE. ElevenLabs scope is frozen to a native speech/transcription provider; voice resources,
  live credential gates, Fal, Replicate, queued media polling, and `prelude::unified` widening stay out
  of this lane.

## ELA-020 — Provider Crate And Speech/Transcription Core

- [x] ELA-020 [owner=worker] [deps=ELA-010] [scope=siumai-provider-elevenlabs,Cargo.toml]
  Goal: Add `siumai-provider-elevenlabs` with provider settings, auth/base URL handling, model constants,
  typed speech/transcription options, error mapping, and no-network client tests for
  `/v1/text-to-speech/{voiceId}` and `/v1/speech-to-text`.
  Validation: `cargo nextest run -p siumai-provider-elevenlabs --features elevenlabs --no-fail-fast`; `cargo fmt --check -p siumai-provider-elevenlabs`.
  Review: review-workstream for provider crate boundary, option mapping, multipart request handling, and voice id behavior.
  Evidence: `EVIDENCE_AND_GATES.md`
  Handoff: DONE. `siumai-provider-elevenlabs` now owns config, auth/base URL/header handling,
  model constants, typed speech/transcription options, request extension traits, speech/transcription
  family model wrappers, and no-network `/v1/text-to-speech/{voiceId}` plus `/v1/speech-to-text`
  request/response tests. Multipart materialization stayed provider-local for this slice; shared
  multipart infrastructure was not required.

## ELA-030 — Registry And Capability Wiring

- [ ] ELA-030 [owner=worker] [deps=ELA-020] [scope=siumai-registry,siumai-core,Cargo.toml]
  Goal: Wire ElevenLabs into feature flags, native metadata, provider catalog, registry factory, and stable
  speech/transcription model handles.
  Validation: `cargo nextest run -p siumai-registry --features elevenlabs elevenlabs --no-fail-fast`; `cargo fmt --check -p siumai-registry`.
  Review: review-workstream for capability metadata, unsupported family rejection, and registry context precedence.
  Evidence: `EVIDENCE_AND_GATES.md`
  Handoff: Registry wiring should mirror Deepgram where possible while preserving ElevenLabs-specific auth and endpoints.

## ELA-040 — Facade, Public Surface, And Examples

- [ ] ELA-040 [owner=worker] [deps=ELA-030] [scope=siumai,siumai/tests,examples]
  Goal: Expose `provider_ext::elevenlabs`, `providers::elevenlabs`, builder/compat helpers, model constants,
  and typed options without widening `prelude::unified`.
  Validation: `cargo nextest run -p siumai --features elevenlabs elevenlabs --no-fail-fast`; targeted public-surface import test; `cargo fmt --check -p siumai`.
  Review: review-workstream for public API shape and export policy.
  Evidence: `EVIDENCE_AND_GATES.md`
  Handoff: Public exports must document the deprecated AI SDK lowercase `elevenlabs` alias decision if Siumai chooses not to mirror it.

## ELA-050 — Closeout

- [ ] ELA-050 [owner=planner] [deps=ELA-020,ELA-030,ELA-040] [scope=docs/workstreams/elevenlabs-audio-provider]
  Goal: Close the ElevenLabs lane or split any residual option/resource gaps into narrow follow-ons.
  Validation: verify-rust-workstream records fresh final gate evidence.
  Review: review-workstream has no blocking findings.
  Evidence: `EVIDENCE_AND_GATES.md`, `WORKSTREAM.json`, `HANDOFF.md`
  Handoff: Summarize shipped behavior, deferred voice/resources work, and next media-provider recommendation.
