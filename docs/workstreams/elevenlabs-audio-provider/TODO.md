# ElevenLabs Audio Provider — TODO

Status: Closed
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

- [x] ELA-030 [owner=worker] [deps=ELA-020] [scope=siumai-registry,siumai-core,siumai-spec,Cargo.toml]
  Goal: Wire ElevenLabs into feature flags, native metadata, provider catalog, registry factory, and stable
  speech/transcription model handles.
  Validation: `cargo nextest run -p siumai-registry --features elevenlabs elevenlabs --no-fail-fast`; `cargo fmt --check -p siumai-registry`.
  Review: review-workstream for capability metadata, unsupported family rejection, and registry context precedence.
  Evidence: `EVIDENCE_AND_GATES.md`
  Handoff: DONE. ElevenLabs is wired into core/registry feature flags, provider ids, native
  metadata, provider catalog, registry factory selection, builder routing, and speech/transcription
  handles. `siumai-spec` was included only to add the legacy `ProviderType::ElevenLabs`
  classification used by provider catalog compatibility metadata. Unsupported non-audio families
  reject before transport use, and registry context overrides are covered by no-network request tests.

## ELA-040 — Facade, Public Surface, And Examples

- [x] ELA-040 [owner=worker] [deps=ELA-030] [scope=siumai,siumai/tests,examples]
  Goal: Expose `provider_ext::elevenlabs`, `providers::elevenlabs`, builder/compat helpers, model constants,
  and typed options without widening `prelude::unified`.
  Validation: `cargo nextest run -p siumai --features elevenlabs elevenlabs --no-fail-fast`; targeted public-surface import test; `cargo fmt --check -p siumai`.
  Review: review-workstream for public API shape and export policy.
  Evidence: `EVIDENCE_AND_GATES.md`
  Handoff: DONE. `siumai::provider_ext::elevenlabs` and `siumai::providers::elevenlabs`
  expose the native audio facade, builder helpers, model constants, typed options, and request
  extension traits. `Provider::elevenlabs()` is available as an explicit compat helper, and
  facade feature/build-time provider accounting works with only `elevenlabs` enabled.
  `prelude::unified` was left unchanged.

## ELA-050 — Closeout

- [x] ELA-050 [owner=planner] [deps=ELA-020,ELA-030,ELA-040] [scope=docs/workstreams/elevenlabs-audio-provider]
  Goal: Close the ElevenLabs lane or split any residual option/resource gaps into narrow follow-ons.
  Validation: verify-rust-workstream records fresh final gate evidence.
  Review: review-workstream has no blocking findings.
  Evidence: `EVIDENCE_AND_GATES.md`, `WORKSTREAM.json`, `HANDOFF.md`
  Handoff: DONE. The ElevenLabs lane is closed after fresh provider-crate, registry, facade,
  formatting, JSON, and whitespace gates. Voice listing/cloning/resources, live credential smoke
  tests, and queued media polling foundations remain separate follow-ons.
