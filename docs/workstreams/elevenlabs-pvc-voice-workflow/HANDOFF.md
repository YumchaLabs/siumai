# ElevenLabs PVC Voice Workflow - Handoff

Status: Active
Last updated: 2026-05-26

## Current State

The workstream is open. EPVC-010 and EPVC-020 are complete. Siumai now exposes PVC metadata and
training JSON resources on `ElevenLabsVoices` with facade exports and focused no-network tests.

## Active Task

- Task ID: EPVC-030
- Owner: worker
- Files: `siumai-provider-elevenlabs/src/providers/elevenlabs/voices.rs`,
  `siumai-provider-elevenlabs/src/providers/elevenlabs/mod.rs`,
  `siumai/src/provider_ext/elevenlabs.rs`,
  `siumai/tests/elevenlabs_voice_resources_public_surface_test.rs`, `CHANGELOG.md`
- Validation: `cargo nextest run -p siumai-provider-elevenlabs --features elevenlabs voices_pvc_samples --no-fail-fast`
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
- `create_pvc_voice`, `update_pvc_voice`, and `train_pvc_voice` are implemented with
  `ElevenLabsCreatePvcVoiceRequest`, `ElevenLabsUpdatePvcVoiceRequest`,
  `ElevenLabsTrainPvcVoiceRequest`, and `ElevenLabsPvcVoiceResponse`.

## Blockers

- None.

## Next Recommended Action

- Implement EPVC-030: PVC sample add/update/delete, sample preview audio, waveform, speaker
  separation status/start, and separated speaker audio resources.
