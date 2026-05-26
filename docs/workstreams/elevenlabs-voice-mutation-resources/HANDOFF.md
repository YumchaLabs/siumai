# ElevenLabs Voice Mutation Resources - Handoff

Status: Active
Last updated: 2026-05-26

## Current State

The workstream is open and scope is frozen. ELVM-010 through ELVM-030 are complete. Siumai now has
provider-owned voice settings get/update plus voice/sample delete resources on the existing
`ElevenLabsVoices` client.

## Active Task

- Task ID: ELVM-040
- Owner: worker
- Files: `siumai-provider-elevenlabs`, `siumai`, `siumai/tests`,
  `docs/workstreams/elevenlabs-voice-mutation-resources`
- Validation: no-network multipart provider tests for accepted IVC/edit endpoints, facade
  public-surface test, and formatting checks.
- Status: READY
- Review: review-workstream for file-part ownership, label encoding, moderation flag semantics, and
  whether edit should split.
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
- Do not implement sample audio until the response shape is re-audited; current docs describe an
  audio endpoint but show an empty JSON response schema.
- Treat PVC as a workflow boundary. PVC create may be a later bounded JSON slice, but training,
  samples, captcha, and manual verification should not be hidden behind a single create call.

## Blockers

- None for ELVM-040.

## Next Recommended Action

- Implement the smallest accepted IVC multipart slice, likely `create_ivc_voice`. Split voice edit
  unless the shared multipart request shape stays bounded.
