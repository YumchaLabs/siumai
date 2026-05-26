# ElevenLabs Voice Mutation Resources - Milestones

Status: Active
Last updated: 2026-05-26

## M0 - Scope Freeze

Status: Complete on 2026-05-26. Evidence is recorded under ELVM-010.

Exit criteria:

- Official voice mutation endpoints are inventoried.
- AI SDK package boundary is checked.
- First implementation slice is chosen.
- PVC/sample/live-credential boundaries are recorded.

Gate:

- Documentation consistency across DESIGN.md, TODO.md, WORKSTREAM.json, HANDOFF.md, and EVIDENCE_AND_GATES.md.

## M1 - Voice Settings JSON

Status: Complete on 2026-05-26. Evidence is recorded in `EVIDENCE_AND_GATES.md` under ELVM-020.

Exit criteria:

- `ElevenLabsVoices::default_settings` exists.
- `ElevenLabsVoices::settings` path-encodes `voice_id`.
- `ElevenLabsVoices::update_settings` serializes typed JSON settings and rejects empty updates.
- No-network tests prove auth/header/base URL reuse, request header merge, body shape, and response
  mapping.
- Facade exports compile.

Gate:

- `cargo nextest run -p siumai-provider-elevenlabs --features elevenlabs voices_settings --no-fail-fast`
- `cargo nextest run -p siumai-provider-elevenlabs --features elevenlabs voices --no-fail-fast`
- `cargo nextest run -p siumai --features elevenlabs elevenlabs_voice_resources --no-fail-fast`
- `cargo fmt --check -p siumai-provider-elevenlabs -p siumai`

## M2 - Delete And Sample Decision

Exit criteria:

- Shared DELETE JSON wiring is implemented or explicitly deferred.
- Voice delete and sample delete are implemented or split with evidence.
- Sample audio response shape is re-audited before implementation.

Gate:

- Focused provider/facade nextest filters for accepted endpoints, or docs-only split validation.
- `git diff --check`

## M3 - IVC Multipart

Exit criteria:

- IVC create is implemented or split with a reason.
- Voice edit is either implemented with the same multipart shape or split.
- Tests prove multipart fields, repeated files, labels, request header merge, and response mapping.

Gate:

- Focused provider/facade nextest filters for accepted multipart endpoints.
- `cargo fmt --check` for touched packages.

## M4 - PVC Boundary

Exit criteria:

- PVC create/update/train/sample/verification workflow boundary is documented.
- At most one bounded PVC slice is implemented unless review accepts a larger coherent slice.
- Live credential tests remain optional and out of required gates unless explicitly accepted.

Gate:

- Official docs re-audit.
- Focused no-network tests for accepted PVC endpoints or docs-only split validation.

## M5 - Closeout

Exit criteria:

- All task ledger items are done, split, or explicitly deferred.
- CHANGELOG records user-visible voice mutation additions.
- Evidence gates are refreshed.
- HANDOFF.md states closure or next executable follow-on.

Gate:

- Provider/facade focused nextest gates for implemented voice mutations.
- `cargo fmt --check` for touched packages.
- `python -m json.tool docs\workstreams\elevenlabs-voice-mutation-resources\WORKSTREAM.json`
- `git diff --check`
