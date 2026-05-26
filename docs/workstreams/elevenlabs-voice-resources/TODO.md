# ElevenLabs Voice Resources — TODO

Status: Active
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

- [ ] ELVR-020 [owner=worker] [deps=ELVR-010] [scope=siumai-provider-elevenlabs,siumai,siumai/tests]
  Goal: Add a provider-owned `ElevenLabsVoices` resource client with typed list/search query,
  paginated voice list response, and voice detail retrieval for `GET /v2/voices` and
  `GET /v1/voices/{voice_id}`.
  Validation: `cargo nextest run -p siumai-provider-elevenlabs --features elevenlabs voices --no-fail-fast`; `cargo nextest run -p siumai --features elevenlabs elevenlabs_voice_resources --no-fail-fast`; `cargo fmt --check -p siumai-provider-elevenlabs -p siumai`.
  Review: review-workstream for resource boundary, query mapping, serde tolerance, auth/header/base URL reuse, and facade export policy.
  Evidence: `EVIDENCE_AND_GATES.md`
  Handoff: Start with no-network tests proving `xi-api-key`, custom headers, `/v2/voices` query
  params, `/v1/voices/{voice_id}` URL encoding, pagination fields, and preservation of unknown
  response metadata.

## ELVR-030 — Pronunciation Dictionary Resource Decision

- [ ] ELVR-030 [owner=planner/worker] [deps=ELVR-020] [scope=siumai-provider-elevenlabs,siumai,siumai/tests,docs/workstreams/elevenlabs-voice-resources]
  Goal: Decide whether to implement pronunciation dictionary resources in this lane or split them,
  then implement the smallest accepted slice.
  Validation: focused provider/facade nextest filter for pronunciation dictionary resource behavior; `cargo fmt --check` for touched packages.
  Review: review-workstream for endpoint count, mutation semantics, and fit with existing TTS locator options.
  Evidence: `EVIDENCE_AND_GATES.md`
  Handoff: Candidate first slice is list/get dictionary metadata; create/update/download/rule
  mutation can split if it would obscure the voice catalog deliverable.

## ELVR-040 — Voice Mutation Split Decision

- [ ] ELVR-040 [owner=planner] [deps=ELVR-020] [scope=docs/workstreams/elevenlabs-voice-resources]
  Goal: Decide whether voice clone/update/delete/settings/sample/PVC APIs belong in this lane or in
  separate mutation-focused workstreams.
  Validation: TODO/MILESTONES/HANDOFF record explicit close-or-split decision.
  Review: review-workstream for scope creep and live credential implications.
  Evidence: `EVIDENCE_AND_GATES.md`
  Handoff: Prefer split unless read-only resources expose an obvious shared helper that makes the
  mutation slice narrow and testable.

## ELVR-050 — Closeout

- [ ] ELVR-050 [owner=planner] [deps=ELVR-020] [scope=docs/workstreams/elevenlabs-voice-resources]
  Goal: Close the lane or split residual dictionary/mutation gaps into narrow follow-ons.
  Validation: verify-rust-workstream records fresh final gate evidence.
  Review: review-workstream has no blocking findings.
  Evidence: `EVIDENCE_AND_GATES.md`, `WORKSTREAM.json`, `HANDOFF.md`
  Handoff: Summarize shipped voice resource behavior, deferred mutation APIs, and whether a generic
  voice management contract is still unjustified.
