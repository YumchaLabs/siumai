# ElevenLabs Pronunciation Dictionary Mutations — Handoff

Status: Active
Last updated: 2026-05-26

## Current State

The workstream is open and scope is frozen. EPDM-010, EPDM-020, and EPDM-030 are complete. Siumai
now has provider-owned create-from-rules and create-from-file pronunciation dictionary mutations.

## Active Task

- Task ID: EPDM-040
- Owner: worker
- Files: `siumai-provider-elevenlabs`, `siumai`, `siumai/tests`
- Validation: focused provider/facade nextest filter for update path encoding, JSON body, and
  metadata response mapping; `cargo fmt --check`
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
- Update metadata and rule mutation can be implemented after create-from-rules shares rule structs.
- Download-by-version is not first because the official `download.mdx` page returned HTTP 500 during
  opening even though `llms.txt` lists it.
- Voice mutation APIs are a separate workstream.

## Blockers

- None for EPDM-020.

## Next Recommended Action

- Use TDD for EPDM-040: first add a no-network provider test for metadata update proving path
  encoding, JSON body shape for `name`/`archived`, empty update rejection, request header merge, and
  dictionary metadata response mapping.
