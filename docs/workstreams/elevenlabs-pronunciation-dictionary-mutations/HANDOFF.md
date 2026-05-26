# ElevenLabs Pronunciation Dictionary Mutations — Handoff

Status: Active
Last updated: 2026-05-26

## Current State

The workstream is open and scope is frozen. EPDM-010 through EPDM-050 are complete. Siumai now has
provider-owned create-from-rules, create-from-file, metadata update, and rule mutation
pronunciation dictionary APIs.

## Active Task

- Task ID: EPDM-060
- Owner: planner/worker
- Files: `siumai-provider-elevenlabs`, `siumai`, `siumai/tests`,
  `docs/workstreams/elevenlabs-pronunciation-dictionary-mutations`
- Validation: official docs accessible or fallback source is recorded; if implemented, focused
  binary GET test proves path/query encoding and bytes mapping.
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
- Rule mutation is implemented for add/set/remove in one slice because all three endpoints share
  version response semantics. Add/set reuse the alias/phoneme rule request struct; remove uses
  `rule_strings`.
- Download-by-version is not first because the official `download.mdx` page returned HTTP 500 during
  opening even though `llms.txt` lists it.
- Voice mutation APIs are a separate workstream.

## Blockers

- None for EPDM-020.

## Next Recommended Action

- Re-audit the official download-by-version docs for EPDM-060. If the endpoint contract is stable,
  implement the binary PLS download with no-network custom transport coverage; otherwise record the
  fallback source and split the download task.
