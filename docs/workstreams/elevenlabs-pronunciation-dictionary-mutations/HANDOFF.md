# ElevenLabs Pronunciation Dictionary Mutations — Handoff

Status: Active
Last updated: 2026-05-26

## Current State

The workstream is open and scope is frozen. EPDM-010 through EPDM-040 are complete. Siumai now has
provider-owned create-from-rules, create-from-file, and metadata update pronunciation dictionary
mutations.

## Active Task

- Task ID: EPDM-050
- Owner: worker
- Files: `siumai-provider-elevenlabs`, `siumai`, `siumai/tests`,
  `docs/workstreams/elevenlabs-pronunciation-dictionary-mutations`
- Validation: focused provider/facade nextest filter for accepted rule mutation endpoints;
  `cargo fmt --check`
- Status: READY
- Review: review-workstream before accepting completion
- Evidence: `docs/workstreams/elevenlabs-pronunciation-dictionary-mutations/EVIDENCE_AND_GATES.md`

## Decisions Since Opening

- Keep mutation/download APIs provider-owned under `provider_ext::elevenlabs::resources`.
- Do not change `ElevenLabsPronunciationDictionaryLocator`; mutation responses should expose IDs and
  version IDs that users can pass into the existing locator.
- Create-from-rules is implemented with typed alias/phoneme request rules and a create response that
  exposes `id`, `version_id`, `version_rules_num`, metadata, and unknown provider fields.
- Create-from-file is implemented with caller-provided bytes plus optional filename, MIME type,
  description, workspace access, and request-level HTTP config. PLS parsing stays out of scope.
- Metadata update is implemented with typed `name`/`archived` fields and local empty-update
  rejection.
- Shared PATCH JSON execution now supports custom transport, preserving no-network resource tests.
- Rule mutation can reuse the create-from-rules alias/phoneme rule request struct.
- Download-by-version is not first because the official `download.mdx` page returned HTTP 500 during
  opening even though `llms.txt` lists it.
- Voice mutation APIs are a separate workstream.

## Blockers

- None for EPDM-020.

## Next Recommended Action

- Use TDD for EPDM-050: first decide whether add/remove/set rules all stay in this lane. The
  smallest cohesive slice is likely `add_rules`, `remove_rules`, and `set_rules` together because
  they share version response semantics and reuse the rule request struct.
