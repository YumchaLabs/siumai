# ElevenLabs PVC Voice Workflow - Handoff

Status: Active
Last updated: 2026-05-26

## Current State

The workstream is open. EPVC-010 through EPVC-030 are complete. Siumai now exposes PVC metadata,
training, sample, and speaker separation resources on `ElevenLabsVoices` with facade exports and
focused no-network tests.

## Active Task

- Task ID: EPVC-040
- Owner: worker
- Files: `siumai-provider-elevenlabs/src/providers/elevenlabs/voices.rs`,
  `siumai-provider-elevenlabs/src/providers/elevenlabs/mod.rs`,
  `siumai/src/provider_ext/elevenlabs.rs`,
  `siumai/tests/elevenlabs_voice_resources_public_surface_test.rs`, `CHANGELOG.md`
- Validation: `cargo nextest run -p siumai-provider-elevenlabs --features elevenlabs voices_pvc_verification --no-fail-fast`
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
- PVC sample/speaker APIs are implemented with `ElevenLabsAddPvcVoiceSamplesRequest`,
  `ElevenLabsUpdatePvcVoiceSampleRequest`, `ElevenLabsPvcVoiceSampleAudioQuery`, typed sample and
  speaker responses, and status response reuse.
- `start_pvc_voice_sample_speaker_separation` sends `{}` through the existing JSON POST helper
  because Siumai's shared resource helper does not currently expose a no-body POST branch.

## Blockers

- None.

## Next Recommended Action

- Implement EPVC-040: PVC manual verification request, captcha get, and captcha recording upload.
