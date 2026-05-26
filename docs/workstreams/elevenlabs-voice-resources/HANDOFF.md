# ElevenLabs Voice Resources — Handoff

Status: Closed
Last updated: 2026-05-26

## Current State

The workstream is closed. ELVR-010 through ELVR-050 are complete. Siumai now has
read-only ElevenLabs voice catalog and pronunciation dictionary metadata resource clients under
provider-owned `resources::*`; mutation-heavy voice APIs are intentionally split.

## Closed Task

- Task ID: ELVR-050
- Owner: planner
- Files: `docs/workstreams/elevenlabs-voice-resources`
- Validation: verify-rust-workstream records fresh final gate evidence
- Status: DONE
- Review: no blocking findings; final gates are recorded in `EVIDENCE_AND_GATES.md`
- Evidence: `docs/workstreams/elevenlabs-voice-resources/EVIDENCE_AND_GATES.md`

## Decisions Since Opening

- AI SDK `@ai-sdk/elevenlabs` has no resource client; Siumai voice resources are an intentional
  provider-owned extension.
- Keep all new APIs under `siumai_provider_elevenlabs::providers::elevenlabs` and facade
  `provider_ext::elevenlabs::resources`.
- Do not add a generic core voice-management trait in this lane.
- Start with read-only `GET /v2/voices` and `GET /v1/voices/{voice_id}`.
- Defer voice cloning, PVC/sample APIs, settings mutation, and live credential smoke tests.
- `ElevenLabsVoices` is provider-owned and exported through facade `resources`, not through
  `prelude::unified`.
- `ElevenLabsPronunciationDictionaries` read-only list/get metadata is narrow enough for this lane
  because it discovers IDs and latest version IDs used by existing TTS pronunciation dictionary
  locators.
- Pronunciation dictionary create/update/rule mutation and PLS download remain split candidates.
- Voice clone/update/delete/settings/sample/PVC APIs are split from this lane because they involve
  multipart uploads, binary retrieval, training/verification workflows, and mutation semantics.

## Blockers

- None.

## Follow-Ons

- Open a mutation-focused workstream for voice clone/update/delete/settings/sample/PVC APIs if those
  are needed.
- Open a separate pronunciation dictionary mutation/download workstream for create/update/rule
  mutation or PLS download.
- Keep `prelude::unified` unchanged unless multiple providers converge on a shared voice-management
  contract.
