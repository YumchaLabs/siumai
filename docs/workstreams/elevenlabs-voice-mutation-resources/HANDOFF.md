# ElevenLabs Voice Mutation Resources - Handoff

Status: Active
Last updated: 2026-05-26

## Current State

The workstream is open and scope is frozen. ELVM-010 and ELVM-020 are complete. Siumai now has
provider-owned voice settings get/update resources on the existing `ElevenLabsVoices` client.

## Active Task

- Task ID: ELVM-030
- Owner: planner/worker
- Files: `siumai-provider-elevenlabs`, `siumai`, `siumai/tests`,
  `docs/workstreams/elevenlabs-voice-mutation-resources`
- Validation: focused provider/facade nextest filter for accepted delete endpoints and
  `git diff --check`.
- Status: READY
- Review: review-workstream for DELETE helper reuse and status response naming.
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
- Do not implement sample audio until the response shape is re-audited; current docs describe an
  audio endpoint but show an empty JSON response schema.
- Treat PVC as a workflow boundary. PVC create may be a later bounded JSON slice, but training,
  samples, captcha, and manual verification should not be hidden behind a single create call.

## Blockers

- None for ELVM-030.

## Next Recommended Action

- Decide whether to add a shared DELETE JSON helper, then implement `delete_voice` and
  `delete_sample` together if the response/status semantics stay aligned.
