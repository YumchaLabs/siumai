# ElevenLabs Pronunciation Dictionary Mutations — Handoff

Status: Active
Last updated: 2026-05-26

## Current State

The workstream is open and scope is frozen. EPDM-010 is complete. The first executable task is a
provider-owned JSON mutation slice for creating pronunciation dictionaries from rules.

## Active Task

- Task ID: EPDM-020
- Owner: worker
- Files: `siumai-provider-elevenlabs`, `siumai`, `siumai/tests`
- Validation: `cargo nextest run -p siumai-provider-elevenlabs --features elevenlabs pronunciation --no-fail-fast`; `cargo nextest run -p siumai --features elevenlabs elevenlabs_voice_resources --no-fail-fast`; `cargo fmt --check -p siumai-provider-elevenlabs -p siumai`
- Status: READY
- Review: review-workstream before accepting completion
- Evidence: `docs/workstreams/elevenlabs-pronunciation-dictionary-mutations/EVIDENCE_AND_GATES.md`

## Decisions Since Opening

- Keep mutation/download APIs provider-owned under `provider_ext::elevenlabs::resources`.
- Do not change `ElevenLabsPronunciationDictionaryLocator`; mutation responses should expose IDs and
  version IDs that users can pass into the existing locator.
- Start with create-from-rules because it is JSON-only and returns the required locator identifiers.
- Create-from-file is a separate multipart slice.
- Update metadata and rule mutation can be implemented after create-from-rules shares rule structs.
- Download-by-version is not first because the official `download.mdx` page returned HTTP 500 during
  opening even though `llms.txt` lists it.
- Voice mutation APIs are a separate workstream.

## Blockers

- None for EPDM-020.

## Next Recommended Action

- Use TDD for EPDM-020: first add a no-network provider test for `create_from_rules`, then implement
  typed request/response structs and facade exports.
