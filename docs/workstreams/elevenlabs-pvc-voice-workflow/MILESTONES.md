# ElevenLabs PVC Voice Workflow - Milestones

Status: Active
Last updated: 2026-05-26

## M0 - Scope And Evidence Freeze

Exit criteria:

- PVC endpoint inventory is explicit.
- Provider-owned resource boundary is explicit.
- Non-goals are explicit, especially live credentials and ordinary voice edit.
- First implementation slice is chosen.

Primary evidence:

- `docs/workstreams/elevenlabs-pvc-voice-workflow/DESIGN.md`
- `docs/workstreams/elevenlabs-pvc-voice-workflow/TODO.md`

## M1 - PVC Metadata And Training

Exit criteria:

- Create/update/train methods are implemented on `ElevenLabsVoices`.
- Request builders cover required fields and documented optional fields.
- No-network tests prove URLs, JSON bodies, headers, and responses.

Primary gates:

- `cargo nextest run -p siumai-provider-elevenlabs --features elevenlabs voices_pvc_metadata --no-fail-fast`
- `cargo nextest run -p siumai --features elevenlabs elevenlabs_voice_resources --no-fail-fast`

## M2 - PVC Samples And Speaker Separation

Exit criteria:

- Sample add/update/delete methods are implemented.
- Sample preview audio, waveform, speaker separation status/start, and separated speaker audio
  methods are implemented.
- No-network tests prove multipart payloads, query parameters, path encoding, and response mapping.

Primary gates:

- `cargo nextest run -p siumai-provider-elevenlabs --features elevenlabs voices_pvc_samples --no-fail-fast`
- `cargo nextest run -p siumai --features elevenlabs elevenlabs_voice_resources --no-fail-fast`

## M3 - PVC Verification

Exit criteria:

- Manual verification request, captcha get, and captcha verify methods are implemented.
- No-network tests cover multipart verification files and captcha recording upload.

Primary gates:

- `cargo nextest run -p siumai-provider-elevenlabs --features elevenlabs voices_pvc_verification --no-fail-fast`
- `cargo nextest run -p siumai --features elevenlabs elevenlabs_voice_resources --no-fail-fast`

## M4 - Closeout

Exit criteria:

- Full focused provider/facade gate set passes.
- `CHANGELOG.md` and workstream evidence reflect the shipped PVC workflow.
- Ordinary voice edit is handed off as the next narrower follow-on.
- `WORKSTREAM.json` status is updated.
