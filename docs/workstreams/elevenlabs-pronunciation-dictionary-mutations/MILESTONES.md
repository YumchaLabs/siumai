# ElevenLabs Pronunciation Dictionary Mutations — Milestones

Status: Active
Last updated: 2026-05-26

## M0 — Scope Freeze

Exit criteria:

- Official mutation/download endpoints are inventoried.
- First implementation slice is chosen.
- Download-by-version docs instability is recorded.
- Voice mutation APIs are excluded from this lane.

Gate:

- Documentation consistency across DESIGN.md, TODO.md, WORKSTREAM.json, HANDOFF.md, and EVIDENCE_AND_GATES.md.

## M1 — Create From Rules

Status: Complete on 2026-05-26. Evidence is recorded in `EVIDENCE_AND_GATES.md` under EPDM-020.

Exit criteria:

- `ElevenLabsPronunciationDictionaries::create_from_rules` exists.
- Alias and phoneme rule request structs serialize to official snake_case JSON.
- Create response maps `id`, `version_id`, `version_rules_num`, metadata fields, and unknown fields.
- No-network tests prove auth/header/base URL reuse and request header merge.
- Facade exports compile.

Gate:

- `cargo nextest run -p siumai-provider-elevenlabs --features elevenlabs pronunciation --no-fail-fast`
- `cargo nextest run -p siumai --features elevenlabs elevenlabs_voice_resources --no-fail-fast`
- `cargo fmt --check -p siumai-provider-elevenlabs -p siumai`

## M2 — Multipart And Metadata Updates

Status: Complete on 2026-05-26. Create-from-file and metadata update are implemented with
no-network evidence recorded under EPDM-030 and EPDM-040.

Exit criteria:

- Create-from-file and/or metadata update are implemented or explicitly split.
- Multipart tests prove form fields and file bytes if create-from-file is accepted.
- PATCH tests prove path encoding, partial body shape, and empty-body rejection if update is accepted.

Gate:

- Focused provider/facade nextest filters for accepted endpoints.
- `cargo fmt --check` for touched packages.

## M3 — Rule Mutation And Download Decision

Status: In progress. Rule mutation completed on 2026-05-26; download-by-version re-audit remains
the next slice.

Exit criteria:

- Add/set/remove rule endpoints are implemented or split with a reason.
- Download-by-version docs are rechecked; binary PLS download is implemented or split.

Gate:

- Focused provider/facade nextest filters for implemented endpoints, or docs-only split validation.

## M4 — Closeout

Exit criteria:

- All task ledger items are done, split, or explicitly deferred.
- CHANGELOG records user-visible mutation/download additions.
- Evidence gates are refreshed.
- HANDOFF.md states closure or next executable follow-on.

Gate:

- Provider/facade focused nextest gates for implemented mutations.
- `cargo fmt --check` for touched packages.
- `python -m json.tool docs\workstreams\elevenlabs-pronunciation-dictionary-mutations\WORKSTREAM.json`
- `git diff --check`
