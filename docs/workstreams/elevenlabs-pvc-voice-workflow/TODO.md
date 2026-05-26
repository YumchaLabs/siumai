# ElevenLabs PVC Voice Workflow - TODO

Status: Active
Last updated: 2026-05-26

## M0 - Scope And Evidence Freeze

- [x] EPVC-010 [owner=planner] [deps=none] [scope=docs/workstreams/elevenlabs-pvc-voice-workflow,official-docs,repo-ref/ai/packages/elevenlabs]
  Goal: Freeze PVC endpoint inventory, provider-owned public-surface boundary, non-goals, and
  validation gates.
  Validation: DESIGN.md, TODO.md, MILESTONES.md, EVIDENCE_AND_GATES.md, WORKSTREAM.json, and
  HANDOFF.md exist and agree.
  Review: planner self-review for workflow boundary, live-credential boundary, and facade export
  policy.
  Evidence: `docs/workstreams/elevenlabs-pvc-voice-workflow/EVIDENCE_AND_GATES.md`
  Handoff: DONE. PVC is a dedicated workflow lane. `edit_voice` remains a separate follow-on.

## M1 - PVC Metadata And Training

- [x] EPVC-020 [owner=worker] [deps=EPVC-010] [scope=siumai-provider-elevenlabs,siumai,siumai/tests]
  Goal: Implement create PVC voice, update PVC voice metadata, and start PVC training with typed
  JSON requests/responses.
  Validation: `cargo nextest run -p siumai-provider-elevenlabs --features elevenlabs voices_pvc_metadata --no-fail-fast`; `cargo nextest run -p siumai --features elevenlabs elevenlabs_voice_resources --no-fail-fast`; `cargo fmt --check -p siumai-provider-elevenlabs -p siumai`.
  Review: review-workstream for required-field validation, documented empty-body behavior, path
  encoding, response naming, and facade export fit.
  Evidence: `EVIDENCE_AND_GATES.md`
  Handoff: DONE. Implemented `create_pvc_voice`, `update_pvc_voice`, and `train_pvc_voice` with
  typed create/update/train requests, shared PVC voice-id response mapping, path encoding,
  per-request HTTP config support, facade exports, changelog coverage, and no-network tests.

## M2 - PVC Samples And Speaker Separation

- [ ] EPVC-030 [owner=worker] [deps=EPVC-020] [scope=siumai-provider-elevenlabs,siumai,siumai/tests]
  Goal: Implement PVC sample add/update/delete, sample audio/waveform retrieval, speaker separation
  status/start, and separated speaker audio retrieval.
  Validation: `cargo nextest run -p siumai-provider-elevenlabs --features elevenlabs voices_pvc_samples --no-fail-fast`; `cargo nextest run -p siumai --features elevenlabs elevenlabs_voice_resources --no-fail-fast`; `cargo fmt --check -p siumai-provider-elevenlabs -p siumai`.
  Review: review-workstream for multipart reuse, sample/speaker path encoding, query encoding,
  response flattening, and binary-vs-base64 response semantics.
  Evidence: `EVIDENCE_AND_GATES.md`
  Handoff: Final status must be DONE, DONE_WITH_CONCERNS, BLOCKED, or NEEDS_CONTEXT.

## M3 - PVC Verification

- [ ] EPVC-040 [owner=worker] [deps=EPVC-030] [scope=siumai-provider-elevenlabs,siumai,siumai/tests]
  Goal: Implement manual verification request, captcha get, and captcha verification upload.
  Validation: `cargo nextest run -p siumai-provider-elevenlabs --features elevenlabs voices_pvc_verification --no-fail-fast`; `cargo nextest run -p siumai --features elevenlabs elevenlabs_voice_resources --no-fail-fast`; `cargo fmt --check -p siumai-provider-elevenlabs -p siumai`.
  Review: review-workstream for multipart field naming, empty captcha response handling, path
  encoding, and status response reuse.
  Evidence: `EVIDENCE_AND_GATES.md`
  Handoff: Final status must be DONE, DONE_WITH_CONCERNS, BLOCKED, or NEEDS_CONTEXT.

## M4 - Closeout

- [ ] EPVC-050 [owner=planner] [deps=EPVC-040] [scope=docs/workstreams/elevenlabs-pvc-voice-workflow,CHANGELOG.md]
  Goal: Close the PVC lane and hand off the ordinary voice edit follow-on.
  Validation: verify-rust-workstream records fresh final gate evidence.
  Review: review-workstream has no blocking findings.
  Evidence: `EVIDENCE_AND_GATES.md`, `WORKSTREAM.json`, `HANDOFF.md`, `CHANGELOG.md`
  Handoff: Summarize remaining risks and the next `edit_voice` task.
