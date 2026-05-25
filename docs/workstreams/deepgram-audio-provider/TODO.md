# Deepgram Audio Provider — TODO

Status: Draft
Last updated: 2026-05-26

## DGA-010 — Scope And Contract Freeze

- [x] DGA-010 [owner=planner] [deps=none] [scope=docs/workstreams/deepgram-audio-provider,repo-ref/ai/packages/deepgram]
  Goal: Freeze the AI SDK package contract, Siumai implementation boundary, non-goals, and validation gates.
  Validation: DESIGN.md, MILESTONES.md, EVIDENCE_AND_GATES.md, WORKSTREAM.json, and HANDOFF.md exist and agree.
  Review: planner self-review for task boundaries before implementation.
  Evidence: `docs/workstreams/deepgram-audio-provider/DESIGN.md`
  Handoff: DONE. Deepgram scope is frozen to a native speech/transcription provider; ElevenLabs, Fal,
  Replicate, live credential gates, and `prelude::unified` widening stay out of this lane.

## DGA-020 — Provider Crate And Speech/Transcription Core

- [x] DGA-020 [owner=worker] [deps=DGA-010] [scope=siumai-provider-deepgram,Cargo.toml]
  Goal: Add `siumai-provider-deepgram` with provider settings, auth/base URL handling, model constants,
  typed speech/transcription options, error mapping, and no-network client tests for `/v1/speak` and
  `/v1/listen`.
  Validation: `cargo nextest run -p siumai-provider-deepgram --features deepgram --no-fail-fast`; `cargo fmt --check -p siumai-provider-deepgram`.
  Review: review-workstream for provider crate boundary and option mapping.
  Evidence: `EVIDENCE_AND_GATES.md`
  Handoff: DONE. `siumai-provider-deepgram` now owns config, auth/base URL/header handling, model
  constants, typed speech/transcription options, speech/transcription family model wrappers, and
  no-network `/v1/speak` plus `/v1/listen` request/response tests. Registry and facade wiring remain
  intentionally out of scope for DGA-030/DGA-040.

## DGA-030 — Registry And Capability Wiring

- [x] DGA-030 [owner=worker] [deps=DGA-020] [scope=siumai-registry,siumai-core,Cargo.toml]
  Goal: Wire Deepgram into feature flags, native metadata, provider catalog, registry factory, and stable
  speech/transcription model handles.
  Validation: `cargo nextest run -p siumai-registry --features deepgram deepgram --no-fail-fast`; `cargo fmt --check -p siumai-registry`.
  Review: review-workstream for capability metadata, unsupported family rejection, and registry context precedence.
  Evidence: `EVIDENCE_AND_GATES.md`
  Handoff: DONE. Registry now exposes Deepgram feature flags, native metadata, provider catalog,
  builder selector, speech/transcription factory paths, provider-specific build overrides, and
  non-audio family rejection before transport use. Facade exports remain intentionally out of scope
  for DGA-040.

## DGA-040 — Facade, Public Surface, And Examples

- [x] DGA-040 [owner=worker] [deps=DGA-030] [scope=siumai,siumai/tests,examples]
  Goal: Expose `provider_ext::deepgram`, `providers::deepgram`, builder/compat helpers, and optional examples
  without widening `prelude::unified`.
  Validation: `cargo nextest run -p siumai --features deepgram deepgram --no-fail-fast`; targeted public-surface import test; `cargo fmt --check -p siumai`.
  Review: review-workstream for public API shape and export policy.
  Evidence: `EVIDENCE_AND_GATES.md`
  Handoff: DONE. The facade now exposes `provider_ext::deepgram`, `providers::deepgram`, model
  constants through `siumai::models::deepgram` and `siumai::constants::deepgram`, and
  `Provider::deepgram()`/`create_deepgram()` helper paths without widening `prelude::unified`.
  No optional example was added in this slice; the public import test covers the new surface.

## DGA-050 — Closeout

- [ ] DGA-050 [owner=planner] [deps=DGA-020,DGA-030,DGA-040] [scope=docs/workstreams/deepgram-audio-provider]
  Goal: Close the Deepgram lane or split any residual option/model gaps into narrow follow-ons.
  Validation: verify-rust-workstream records fresh final gate evidence.
  Review: review-workstream has no blocking findings.
  Evidence: `EVIDENCE_AND_GATES.md`, `WORKSTREAM.json`, `HANDOFF.md`
  Handoff: Summarize shipped behavior, unsupported families, intentional divergences, and follow-ons.
