# Deepgram Audio Provider — Handoff

Status: Draft
Last updated: 2026-05-26

## Current State

The workstream is open. DGA-010 has frozen the execution lane from the closed AI SDK provider market expansion
decision. Implementation has not started.

## Active Task

- Task ID: DGA-020
- Owner: worker
- Files:
  - `siumai-provider-deepgram/*`
  - `Cargo.toml`
- Validation:
  - `cargo nextest run -p siumai-provider-deepgram --features deepgram --no-fail-fast`
  - `cargo fmt --check -p siumai-provider-deepgram`
- Status: READY_TO_IMPLEMENT
- Review: Pending after implementation
- Evidence: `docs/workstreams/deepgram-audio-provider/EVIDENCE_AND_GATES.md`

## Decisions Since Last Update

- Deepgram is a native audio provider crate candidate, not an OpenAI-compatible preset.
- The first lane scope is speech and transcription only.
- Use `DEEPGRAM_API_KEY` fallback and `authorization: Token <key>` to match AI SDK.
- Expose provider-owned typed options and model constants under a stable Deepgram provider root.
- Do not widen `prelude::unified`.
- Do not include ElevenLabs, Replicate, Fal, or common media polling in this lane.

## Blockers

- None currently.

## Next Recommended Action

- Start DGA-020 by adding `siumai-provider-deepgram` with no-network request/response tests for `/v1/speak`
  and `/v1/listen`.
