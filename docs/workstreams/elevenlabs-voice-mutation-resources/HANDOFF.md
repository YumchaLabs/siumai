# ElevenLabs Voice Mutation Resources - Handoff

Status: Closed
Last updated: 2026-05-26

## Current State

The workstream is closed. ELVM-010 through ELVM-060 are complete. Siumai now has provider-owned
voice settings get/update, voice/sample delete, and IVC create resources on the existing
`ElevenLabsVoices` client.

## Active Task

- Task ID: none
- Owner: none
- Files: none
- Validation: final gates recorded in `EVIDENCE_AND_GATES.md`.
- Status: CLOSED
- Review: closeout review found no blocking workstream or code-quality findings.
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
- PVC create/update/train/sample/verification is split into a dedicated follow-on workflow lane.

## Blockers

- None remaining for this lane.

## Next Recommended Action

- Open a dedicated PVC workflow workstream if PVC support is needed.
- Open a smaller voice edit or sample-audio follow-on if those endpoints become priority.
