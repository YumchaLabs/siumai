# ElevenLabs Voice Mutation Resources - Handoff

Status: Active
Last updated: 2026-05-26

## Current State

The workstream is open and scope is frozen. ELVM-010 through ELVM-040 are complete. Siumai now has
provider-owned voice settings get/update, voice/sample delete, and IVC create resources on the
existing `ElevenLabsVoices` client.

## Active Task

- Task ID: ELVM-050
- Owner: planner/worker
- Files: `docs/workstreams/elevenlabs-voice-mutation-resources`, `siumai-provider-elevenlabs`,
  `siumai`, `siumai/tests`
- Validation: official docs re-audit, no-network JSON/multipart tests for accepted PVC endpoint, or
  docs-only split validation.
- Status: READY
- Review: review-workstream for workflow boundary, live-credential boundary, and response shape.
- Evidence: `docs/workstreams/elevenlabs-voice-mutation-resources/EVIDENCE_AND_GATES.md`

## Decisions Since Opening

- AI SDK `@ai-sdk/elevenlabs` has no voice resource client; Siumai voice mutation resources are an
  intentional provider-owned extension.
- Keep all new APIs under `provider_ext::elevenlabs::resources::*` and
  `providers::elevenlabs::resources::*`.
- Do not add a generic voice-management trait or widen `prelude::unified`.
- Start with settings get/update because the official API is JSON, bounded, and reuses existing
  settings response semantics.
- Settings get/update is implemented with `default_settings`, `settings`, `update_settings`,
  `ElevenLabsUpdateVoiceSettingsRequest`, and `ElevenLabsVoiceSettingsUpdateResponse`.
- Voice and sample deletion are implemented with shared `ElevenLabsVoiceStatusResponse`.
- Core DELETE JSON execution now supports custom transports, preserving no-network resource tests.
- IVC create is implemented with `create_ivc_voice`, `ElevenLabsCreateIvcVoiceRequest`,
  `ElevenLabsCreateIvcVoiceResponse`, and reusable `ElevenLabsVoiceSampleFile`.
- Voice edit remains split from IVC create because it has a path parameter, required `name`, optional
  new files, `moderate_metadata`, and a status response rather than `voice_id`.
- Do not implement sample audio until the response shape is re-audited; current docs describe an
  audio endpoint but show an empty JSON response schema.
- Treat PVC as a workflow boundary. PVC create may be a later bounded JSON slice, but training,
  samples, captcha, and manual verification should not be hidden behind a single create call.

## Blockers

- None for ELVM-050.

## Next Recommended Action

- Decide whether PVC create belongs in this lane as a bounded JSON slice or whether PVC should split
  into a dedicated workflow workstream covering create/update/train/samples/verification.
