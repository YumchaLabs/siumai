# ElevenLabs Voice Resources — Handoff

Status: Active
Last updated: 2026-05-26

## Current State

The workstream is open and scope is frozen. ELVR-010 through ELVR-030 are complete. Siumai now has
read-only ElevenLabs voice catalog and pronunciation dictionary metadata resource clients under
provider-owned `resources::*`.

## Active Task

- Task ID: ELVR-040
- Owner: planner
- Files: `docs/workstreams/elevenlabs-voice-resources`
- Validation: TODO/MILESTONES/HANDOFF record explicit close-or-split decision
- Status: READY
- Review: check scope creep and live credential implications before accepting voice mutation work
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

## Blockers

- None for ELVR-020.

## Next Recommended Action

- Review ELVR-040 and decide whether voice clone/update/delete/settings/sample/PVC APIs should stay
  split from this read-only resource lane.
