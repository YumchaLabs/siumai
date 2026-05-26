# ElevenLabs PVC Voice Workflow - Handoff

Status: Active
Last updated: 2026-05-26

## Current State

The workstream is open. Scope, endpoint inventory, non-goals, and gate set are frozen in the
authoritative docs. No PVC implementation has landed yet in this lane.

## Active Task

- Task ID: EPVC-020
- Owner: worker
- Files: `siumai-provider-elevenlabs/src/providers/elevenlabs/voices.rs`,
  `siumai-provider-elevenlabs/src/providers/elevenlabs/mod.rs`,
  `siumai/src/provider_ext/elevenlabs.rs`,
  `siumai/tests/elevenlabs_voice_resources_public_surface_test.rs`, `CHANGELOG.md`
- Validation: `cargo nextest run -p siumai-provider-elevenlabs --features elevenlabs voices_pvc_metadata --no-fail-fast`
- Status: NEEDS_CONTEXT
- Review: pending
- Evidence: `docs/workstreams/elevenlabs-pvc-voice-workflow/EVIDENCE_AND_GATES.md`

## Decisions Since Opening

- PVC stays under the existing provider-owned `ElevenLabsVoices` resource client.
- PVC public exports stay under `provider_ext::elevenlabs::resources::*` and
  `providers::elevenlabs::resources::*`.
- `prelude::unified` is not widened.
- Live credential tests are out of scope; custom transport tests are required.
- Ordinary voice edit remains a separate follow-on after PVC lane closeout.

## Blockers

- None.

## Next Recommended Action

- Implement EPVC-020: PVC create/update/train JSON methods, request/response types, facade exports,
  changelog entry, and focused no-network tests.
