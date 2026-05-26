# ElevenLabs Voice Resources — Milestones

Status: Active
Last updated: 2026-05-26

## M0 — Scope Freeze

Exit criteria:

- AI SDK package surface and official ElevenLabs resource APIs are inventoried.
- The lane explicitly keeps resource clients provider-owned under `resources::*`.
- The first implementation task is read-only voice catalog list/get.
- Voice mutation, live credential gates, and `prelude::unified` widening are excluded from the first task.

Gate:

- Documentation consistency across DESIGN.md, TODO.md, WORKSTREAM.json, HANDOFF.md, and EVIDENCE_AND_GATES.md.

## M1 — Voice Catalog Resource

Status: Complete.

Exit criteria:

- `ElevenLabsVoices` exists in `siumai-provider-elevenlabs`.
- Voice list query maps documented `GET /v2/voices` parameters including pagination/filter fields.
- Voice detail maps `GET /v1/voices/{voice_id}` with URL encoding and configured auth/header/base URL behavior.
- Response structs preserve documented fields and tolerate unknown provider fields.
- Facade exports compile through `provider_ext::elevenlabs::resources`.

Gate:

- `cargo nextest run -p siumai-provider-elevenlabs --features elevenlabs voices --no-fail-fast`
- `cargo nextest run -p siumai --features elevenlabs elevenlabs_voice_resources --no-fail-fast`
- `cargo fmt --check -p siumai-provider-elevenlabs -p siumai`

## M2 — Dictionary Or Split

Status: Complete.

Exit criteria:

- Pronunciation dictionary resources are either implemented in a bounded slice or split into a documented follow-on.
- The decision explains interaction with existing TTS pronunciation dictionary locator options.

Gate:

- Focused nextest filter for implemented dictionary resources, or docs-only close/split validation.

## M3 — Closeout

Status: Ready after ELVR-040.

Exit criteria:

- All task ledger items are done, split, or explicitly deferred.
- Evidence gates are refreshed.
- HANDOFF.md states closure or next executable task.

Gate:

- Provider/facade focused nextest gates for implemented resources.
- `cargo fmt --check` for touched packages.
- `python -m json.tool docs\workstreams\elevenlabs-voice-resources\WORKSTREAM.json`
- `git diff --check`
