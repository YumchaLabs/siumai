# ElevenLabs PVC Voice Workflow - Evidence And Gates

Status: Active
Last updated: 2026-05-26

## Smallest Current Repro

```bash
cargo nextest run -p siumai-provider-elevenlabs --features elevenlabs voices_pvc_metadata --no-fail-fast
```

This will prove the first PVC slice once EPVC-020 lands.

## Gate Set

### Reference And Boundary Gate

```bash
python .agents\skills\siumai-ai-sdk-maintenance\scripts\resolve_ai_sdk_repo.py
```

Confirms the local AI SDK reference checkout is available before public-surface decisions.

### Targeted Iteration Gates

```bash
cargo nextest run -p siumai-provider-elevenlabs --features elevenlabs voices_pvc_metadata --no-fail-fast
cargo nextest run -p siumai-provider-elevenlabs --features elevenlabs voices_pvc_samples --no-fail-fast
cargo nextest run -p siumai-provider-elevenlabs --features elevenlabs voices_pvc_verification --no-fail-fast
```

These gates prove the three PVC workflow slices independently.

### Package And Facade Gates

```bash
cargo nextest run -p siumai-provider-elevenlabs --features elevenlabs voices --no-fail-fast
cargo nextest run -p siumai --features elevenlabs elevenlabs_voice_resources --no-fail-fast
```

These gates prove the broader ElevenLabs voice resource behavior and facade export surface.

### Formatting And Hygiene Gates

```bash
cargo fmt --check -p siumai-provider-elevenlabs -p siumai
python -m json.tool docs\workstreams\elevenlabs-pvc-voice-workflow\WORKSTREAM.json
git diff --check
```

These gates prove formatting, machine-readable workstream metadata, and whitespace hygiene.

### Review Gate

Run `review-workstream` before accepting task or lane completion. Record blocking findings, missing
gates, and residual risks here or in the closeout notes.

## Evidence Anchors

- `docs/workstreams/elevenlabs-pvc-voice-workflow/DESIGN.md`
- `docs/workstreams/elevenlabs-pvc-voice-workflow/TODO.md`
- `docs/workstreams/elevenlabs-pvc-voice-workflow/MILESTONES.md`
- `repo-ref/ai/packages/elevenlabs`
- `https://elevenlabs.io/docs/llms.txt`
- `https://elevenlabs.io/docs/api-reference/voices/pvc/create.mdx`
- `https://elevenlabs.io/docs/api-reference/voices/pvc/update.mdx`
- `https://elevenlabs.io/docs/api-reference/voices/pvc/train.mdx`
- `https://elevenlabs.io/docs/api-reference/voices/pvc/samples/create.mdx`
- `https://elevenlabs.io/docs/api-reference/voices/pvc/samples/update.mdx`
- `https://elevenlabs.io/docs/api-reference/voices/pvc/samples/delete.mdx`
- `https://elevenlabs.io/docs/api-reference/voices/pvc/samples/get-audio.mdx`
- `https://elevenlabs.io/docs/api-reference/voices/pvc/samples/get-waveform.mdx`
- `https://elevenlabs.io/docs/api-reference/voices/pvc/samples/get-speaker-separation-status.mdx`
- `https://elevenlabs.io/docs/api-reference/voices/pvc/samples/separate-speakers.mdx`
- `https://elevenlabs.io/docs/api-reference/voices/pvc/samples/get-separated-speaker-audio.mdx`
- `https://elevenlabs.io/docs/api-reference/voices/pvc/verification/request.mdx`
- `https://elevenlabs.io/docs/api-reference/voices/pvc/verification/captcha.mdx`
- `https://elevenlabs.io/docs/api-reference/voices/pvc/verification/captcha/verify.mdx`

## Fresh Runs

- 2026-05-26: `cargo nextest run -p siumai-provider-elevenlabs --features elevenlabs voices_pvc_metadata --no-fail-fast`
  - Result: PASS. 3 tests passed, 26 skipped.
  - Covers: PVC create/update/train JSON URLs, request bodies, headers, response mapping, and
    required create metadata validation.
- 2026-05-26: `cargo nextest run -p siumai --features elevenlabs elevenlabs_voice_resources --no-fail-fast`
  - Result: PASS. 1 test passed, 546 skipped across 213 binaries.
  - Covers: facade exports for the new PVC request/response types under both
    `provider_ext::elevenlabs::resources` and `providers::elevenlabs::resources`.
- 2026-05-26: `cargo fmt --check -p siumai-provider-elevenlabs -p siumai`
  - Result: PASS.
  - Covers: formatting for the crates touched by EPVC-020.
- 2026-05-26: `cargo nextest run -p siumai-provider-elevenlabs --features elevenlabs voices_pvc_samples --no-fail-fast`
  - Result: PASS. 3 tests passed, 29 skipped.
  - Covers: PVC sample multipart upload, sample update/delete, preview audio query parameters,
    waveform retrieval, speaker separation status/start, separated speaker audio retrieval, path
    encoding, response flattening, and status mapping.
- 2026-05-26: `cargo nextest run -p siumai --features elevenlabs elevenlabs_voice_resources --no-fail-fast`
  - Result: PASS. 1 test passed, 546 skipped across 213 binaries.
  - Covers: facade exports for the new PVC sample and speaker separation request/response types.
- 2026-05-26: `cargo fmt --check -p siumai-provider-elevenlabs -p siumai`
  - Result: PASS.
  - Covers: formatting for the crates touched by EPVC-030.

## Notes

Fresh verification is required before marking EPVC-020, EPVC-030, EPVC-040, EPVC-050, or the goal
complete.
