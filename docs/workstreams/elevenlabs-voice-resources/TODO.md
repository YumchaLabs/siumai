# ElevenLabs Voice Resources — TODO

Status: Closed
Last updated: 2026-05-26

## ELVR-010 — Scope And Resource Contract Freeze

- [x] ELVR-010 [owner=planner] [deps=none] [scope=docs/workstreams/elevenlabs-voice-resources,repo-ref/ai/packages/elevenlabs]
  Goal: Freeze the resource boundary, provider-owned public surface, first implementation slice,
  non-goals, and validation gates.
  Validation: DESIGN.md, MILESTONES.md, EVIDENCE_AND_GATES.md, WORKSTREAM.json, and HANDOFF.md exist and agree.
  Review: planner self-review for resource/task boundaries before implementation.
  Evidence: `docs/workstreams/elevenlabs-voice-resources/DESIGN.md`
  Handoff: DONE. Scope is provider-owned `resources::*`; first implementation is read-only voice
  catalog list/get. Voice cloning, PVC/sample mutation, live credentials, and `prelude::unified`
  widening stay out of the first task.

## ELVR-020 — Read-Only Voice Catalog Resource

- [x] ELVR-020 [owner=worker] [deps=ELVR-010] [scope=siumai-provider-elevenlabs,siumai,siumai/tests]
  Goal: Add a provider-owned `ElevenLabsVoices` resource client with typed list/search query,
  paginated voice list response, and voice detail retrieval for `GET /v2/voices` and
  `GET /v1/voices/{voice_id}`.
  Validation: `cargo nextest run -p siumai-provider-elevenlabs --features elevenlabs voices --no-fail-fast`; `cargo nextest run -p siumai --features elevenlabs elevenlabs_voice_resources --no-fail-fast`; `cargo fmt --check -p siumai-provider-elevenlabs -p siumai`.
  Review: review-workstream for resource boundary, query mapping, serde tolerance, auth/header/base URL reuse, and facade export policy.
  Evidence: `EVIDENCE_AND_GATES.md`
  Handoff: DONE. `ElevenLabsClient::voices()` now returns `ElevenLabsVoices`; list/get resources
  are exported through provider crate and facade `provider_ext::elevenlabs::resources`. No-network
  tests prove `xi-api-key`, explicit key override, custom/global/request header merge, base URL
  override, documented `/v2/voices` query params, `/v1/voices/{voice_id}` URL encoding, stable field
  mapping, unknown metadata preservation, and facade imports.

## ELVR-030 — Pronunciation Dictionary Resource Decision

- [x] ELVR-030 [owner=planner/worker] [deps=ELVR-020] [scope=siumai-provider-elevenlabs,siumai,siumai/tests,docs/workstreams/elevenlabs-voice-resources]
  Goal: Decide whether to implement pronunciation dictionary resources in this lane or split them,
  then implement the smallest accepted slice.
  Validation: focused provider/facade nextest filter for pronunciation dictionary resource behavior; `cargo fmt --check` for touched packages.
  Review: review-workstream for endpoint count, mutation semantics, and fit with existing TTS locator options.
  Evidence: `EVIDENCE_AND_GATES.md`
  Handoff: DONE. Implemented provider-owned `ElevenLabsPronunciationDictionaries` read-only
  metadata resources for `GET /v1/pronunciation-dictionaries` and
  `GET /v1/pronunciation-dictionaries/{pronunciation_dictionary_id}`. This exposes dictionary IDs,
  latest version IDs, metadata, and detail rules needed to discover inputs for existing TTS
  pronunciation dictionary locators. Create/update/rule mutation and PLS download remain split
  candidates.

## ELVR-040 — Voice Mutation Split Decision

- [x] ELVR-040 [owner=planner] [deps=ELVR-020] [scope=docs/workstreams/elevenlabs-voice-resources]
  Goal: Decide whether voice clone/update/delete/settings/sample/PVC APIs belong in this lane or in
  separate mutation-focused workstreams.
  Validation: TODO/MILESTONES/HANDOFF record explicit close-or-split decision.
  Review: review-workstream for scope creep and live credential implications.
  Evidence: `EVIDENCE_AND_GATES.md`
  Handoff: DONE. Split voice clone/update/delete/settings/sample/PVC APIs into future
  mutation-focused workstreams. Official endpoint inventory shows multipart uploads, binary sample
  retrieval, training/verification workflows, delete/update semantics, and likely live credential
  smoke tests; that would obscure the read-only catalog/dictionary deliverable.

## ELVR-050 — Closeout

- [x] ELVR-050 [owner=planner] [deps=ELVR-020] [scope=docs/workstreams/elevenlabs-voice-resources]
  Goal: Close the lane or split residual dictionary/mutation gaps into narrow follow-ons.
  Validation: verify-rust-workstream records fresh final gate evidence.
  Review: review-workstream has no blocking findings.
  Evidence: `EVIDENCE_AND_GATES.md`, `WORKSTREAM.json`, `HANDOFF.md`
  Handoff: DONE. Lane closed with read-only voice catalog and pronunciation dictionary metadata
  resources shipped. Voice mutation/PVC/sample/settings APIs and pronunciation dictionary
  mutation/download APIs are follow-ons. A generic voice-management contract remains unjustified
  because only ElevenLabs has this provider-specific resource shape in Siumai today.
