# ElevenLabs Voice Mutation Resources - TODO

Status: Closed
Last updated: 2026-05-26

## ELVM-010 - Scope And Endpoint Contract Freeze

- [x] ELVM-010 [owner=planner] [deps=none] [scope=docs/workstreams/elevenlabs-voice-mutation-resources,official-docs,repo-ref/ai/packages/elevenlabs]
  Goal: Freeze the endpoint inventory, first mutation slice, provider-owned boundary, and validation gates.
  Validation: DESIGN.md, TODO.md, MILESTONES.md, EVIDENCE_AND_GATES.md, WORKSTREAM.json, and HANDOFF.md exist and agree.
  Review: planner self-review for voice mutation/PVC/live-credential boundaries.
  Evidence: `docs/workstreams/elevenlabs-voice-mutation-resources/EVIDENCE_AND_GATES.md`
  Handoff: DONE. AI SDK has no voice resource client; Siumai keeps this provider-owned under
  `resources::*`. First executable slice is voice settings get/update.

## ELVM-020 - Voice Settings Get And Update

- [x] ELVM-020 [owner=worker] [deps=ELVM-010] [scope=siumai-provider-elevenlabs,siumai,siumai/tests]
  Goal: Add `default_settings`, `settings`, and `update_settings` support to `ElevenLabsVoices` for
  the documented settings endpoints.
  Validation: `cargo nextest run -p siumai-provider-elevenlabs --features elevenlabs voices_settings --no-fail-fast`; `cargo nextest run -p siumai-provider-elevenlabs --features elevenlabs voices --no-fail-fast`; `cargo nextest run -p siumai --features elevenlabs elevenlabs_voice_resources --no-fail-fast`; `cargo fmt --check -p siumai-provider-elevenlabs -p siumai`.
  Review: review-workstream for request naming, empty-update behavior, path encoding, and facade
  export fit.
  Evidence: `EVIDENCE_AND_GATES.md`
  Handoff: DONE. Implemented `default_settings`, `settings`, and `update_settings` with typed
  `ElevenLabsUpdateVoiceSettingsRequest`, status response mapping, path encoding, header merge,
  empty-update rejection, no-network provider tests, facade exports, and CHANGELOG coverage.

## ELVM-030 - Delete Voice And Sample Delete Decision

- [x] ELVM-030 [owner=planner/worker] [deps=ELVM-020] [scope=siumai-core,siumai-provider-elevenlabs,siumai,siumai/tests,docs/workstreams/elevenlabs-voice-mutation-resources]
  Goal: Decide whether to add a shared DELETE JSON helper and implement `delete_voice` plus
  `delete_sample`, or split one endpoint if response/error semantics differ.
  Validation: focused provider/facade nextest filter for accepted delete endpoints; `git diff --check`.
  Review: review-workstream for DELETE helper reuse and status response naming.
  Evidence: `EVIDENCE_AND_GATES.md`
  Handoff: DONE. Core already had DELETE JSON execution but lacked custom transport support. Added
  custom transport DELETE support, provider-owned `execute_delete_json`, `delete_voice`,
  `delete_sample`, shared `ElevenLabsVoiceStatusResponse`, no-network tests, facade exports, and
  CHANGELOG coverage.

## ELVM-040 - IVC Create And Voice Edit Multipart

- [x] ELVM-040 [owner=worker] [deps=ELVM-020] [scope=siumai-provider-elevenlabs,siumai,siumai/tests]
  Goal: Implement the smallest accepted multipart IVC/edit slice, likely `create_ivc_voice` first
  and `edit_voice` if the shared request shape stays bounded.
  Validation: no-network multipart provider tests for files, labels, optional fields, path encoding,
  header merge, and response mapping; facade public-surface compile test.
  Review: review-workstream for file-part ownership, label encoding, moderation flag semantics, and
  whether edit should split.
  Evidence: `EVIDENCE_AND_GATES.md`
  Handoff: DONE. Implemented `create_ivc_voice` with typed `ElevenLabsCreateIvcVoiceRequest`,
  reusable `ElevenLabsVoiceSampleFile`, multipart multi-sample upload, labels serialization,
  response mapping, no-network tests, facade exports, and CHANGELOG coverage. `edit_voice` remains
  split because it has different path/body requirements and a status response.

## ELVM-050 - PVC Boundary And First PVC Slice

- [x] ELVM-050 [owner=planner/worker] [deps=ELVM-020] [scope=docs/workstreams/elevenlabs-voice-mutation-resources,siumai-provider-elevenlabs,siumai]
  Goal: Decide whether PVC create/update/train/sample/verification belongs in this lane or should
  split into a PVC workflow lane; implement at most one bounded first PVC slice if accepted.
  Validation: official docs re-audit, no-network JSON/multipart tests for accepted PVC endpoint, or
  docs-only split validation.
  Review: review-workstream for workflow boundary, live-credential boundary, and response shape.
  Evidence: `EVIDENCE_AND_GATES.md`
  Handoff: DONE. Split PVC create/update/train/sample/verification into a dedicated follow-on
  workflow lane because PVC operations share training, sample, captcha, and manual verification
  state that should be designed together.

## ELVM-060 - Closeout

- [x] ELVM-060 [owner=planner] [deps=ELVM-020] [scope=docs/workstreams/elevenlabs-voice-mutation-resources,CHANGELOG.md]
  Goal: Close the lane or split residual voice mutation/PVC/sample gaps into narrow follow-ons.
  Validation: verify-rust-workstream records fresh final gate evidence.
  Review: review-workstream has no blocking findings.
  Evidence: `EVIDENCE_AND_GATES.md`, `WORKSTREAM.json`, `HANDOFF.md`, `CHANGELOG.md`
  Handoff: DONE. Closed this lane after implementing voice settings get/update, voice/sample delete,
  and IVC create. PVC workflow APIs, voice edit, and sample audio remain explicit follow-ons.
