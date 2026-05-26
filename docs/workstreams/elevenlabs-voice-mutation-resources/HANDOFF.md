# ElevenLabs Voice Mutation Resources - Handoff

Status: Active
Last updated: 2026-05-26

## Current State

The workstream is open and scope is frozen. ELVM-010 is complete. The official ElevenLabs voice
mutation inventory has been audited enough to choose the first implementation slice: voice settings
get/update on the existing provider-owned `ElevenLabsVoices` resource client.

## Active Task

- Task ID: ELVM-020
- Owner: worker
- Files: `siumai-provider-elevenlabs`, `siumai`, `siumai/tests`
- Validation: focused provider settings tests, provider voices tests, facade public-surface test,
  and formatting checks.
- Status: READY
- Review: review-workstream for request naming, empty-update behavior, path encoding, and facade fit.
- Evidence: `docs/workstreams/elevenlabs-voice-mutation-resources/EVIDENCE_AND_GATES.md`

## Decisions Since Opening

- AI SDK `@ai-sdk/elevenlabs` has no voice resource client; Siumai voice mutation resources are an
  intentional provider-owned extension.
- Keep all new APIs under `provider_ext::elevenlabs::resources::*` and
  `providers::elevenlabs::resources::*`.
- Do not add a generic voice-management trait or widen `prelude::unified`.
- Start with settings get/update because the official API is JSON, bounded, and reuses existing
  settings response semantics.
- Do not implement sample audio until the response shape is re-audited; current docs describe an
  audio endpoint but show an empty JSON response schema.
- Treat PVC as a workflow boundary. PVC create may be a later bounded JSON slice, but training,
  samples, captcha, and manual verification should not be hidden behind a single create call.

## Blockers

- None for ELVM-020.

## Next Recommended Action

- Implement ELVM-020 with TDD: no-network tests first for `default_settings`, `settings`, and
  `update_settings`, then facade exports and focused gates.
