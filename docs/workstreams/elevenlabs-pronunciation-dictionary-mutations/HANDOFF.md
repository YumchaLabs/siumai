# ElevenLabs Pronunciation Dictionary Mutations — Handoff

Status: Active
Last updated: 2026-05-26

## Current State

The workstream is open and scope is frozen. EPDM-010 through EPDM-060 are complete. Siumai now has
provider-owned create-from-rules, create-from-file, metadata update, and rule mutation
pronunciation dictionary APIs, plus binary PLS download by dictionary/version id.

## Active Task

- Task ID: EPDM-070
- Owner: planner
- Files: `docs/workstreams/elevenlabs-pronunciation-dictionary-mutations`, `CHANGELOG.md`
- Validation: verify-rust-workstream records fresh final gate evidence.
- Status: READY
- Review: review-workstream has no blocking findings before closure.
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
- Download-by-version was re-audited after the opening transient HTTP 500. The official
  `download.mdx` page is accessible and implemented as binary PLS download.
- Voice mutation APIs are a separate workstream.

## Blockers

- None for EPDM-020.

## Next Recommended Action

- Run closeout/review for EPDM-070. If accepted, close this pronunciation dictionary mutation lane
  and open or resume the separate ElevenLabs voice mutation workstream.
