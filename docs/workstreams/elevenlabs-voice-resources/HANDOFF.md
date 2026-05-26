# ElevenLabs Voice Resources — Handoff

Status: Active
Last updated: 2026-05-26

## Current State

The workstream is open and scope is frozen. ELVR-010 and ELVR-020 are complete. Siumai now has a
read-only ElevenLabs voice catalog resource client under provider-owned `resources::*`.

## Active Task

- Task ID: ELVR-030
- Owner: planner/worker
- Files: `siumai-provider-elevenlabs`, `siumai`, `siumai/tests`, `docs/workstreams/elevenlabs-voice-resources`
- Validation: focused provider/facade nextest filter for any accepted pronunciation dictionary resource slice; `cargo fmt --check` for touched packages
- Status: READY
- Review: decide implementation vs split before adding dictionary endpoints
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
- Decide pronunciation dictionary resources after voice catalog lands.

## Blockers

- None for ELVR-020.

## Next Recommended Action

- Review ELVR-030 and decide whether pronunciation dictionary list/get metadata is narrow enough for
  this lane or should split into a follow-on.
