# ElevenLabs Voice Resources — Handoff

Status: Active
Last updated: 2026-05-26

## Current State

The workstream is open and scope is frozen. ELVR-010 is complete. The first implementation task is a
read-only ElevenLabs voice catalog resource client under provider-owned `resources::*`.

## Active Task

- Task ID: ELVR-020
- Owner: worker
- Files: `siumai-provider-elevenlabs`, `siumai`, `siumai/tests`
- Validation: `cargo nextest run -p siumai-provider-elevenlabs --features elevenlabs voices --no-fail-fast`; `cargo nextest run -p siumai --features elevenlabs elevenlabs_voice_resources --no-fail-fast`; `cargo fmt --check -p siumai-provider-elevenlabs -p siumai`
- Status: READY
- Review: review-workstream before accepting completion
- Evidence: `docs/workstreams/elevenlabs-voice-resources/EVIDENCE_AND_GATES.md`

## Decisions Since Opening

- AI SDK `@ai-sdk/elevenlabs` has no resource client; Siumai voice resources are an intentional
  provider-owned extension.
- Keep all new APIs under `siumai_provider_elevenlabs::providers::elevenlabs` and facade
  `provider_ext::elevenlabs::resources`.
- Do not add a generic core voice-management trait in this lane.
- Start with read-only `GET /v2/voices` and `GET /v1/voices/{voice_id}`.
- Defer voice cloning, PVC/sample APIs, settings mutation, and live credential smoke tests.
- Decide pronunciation dictionary resources after voice catalog lands.

## Blockers

- None for ELVR-020.

## Next Recommended Action

- Start ELVR-020 with tests for `ElevenLabsVoices` list/get request construction and response
  mapping, then expose the resource client through `provider_ext::elevenlabs::resources`.
