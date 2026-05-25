# Provider Model Catalog AI SDK Refresh — Handoff

Status: Closed
Last updated: 2026-05-25

## Current State

Workstream closed after the local AI SDK model catalog audit and focused Rust gates passed for the low-risk additive refresh.

## Active Task

- Task ID: PMCA-040
- Owner: codex
- Files: provider model constants, facade model exports, registry catalog tests, workstream docs
- Validation: audit script plus focused Rust tests
- Status: COMPLETE
- Review: no blocking findings from focused gates
- Evidence: EVIDENCE_AND_GATES.md

## Decisions Since Last Update

- DeepInfra is deferred as a policy decision because the mismatch is large and existing docs describe a curated subset.
- The completed implementation slice covers low-risk additive model constants only.
- Model catalog refresh preserves existing `popular`/representative model constants; new ids are added to model sets and capability/catalog paths without changing those recommendations.
- `siumai::provider_ext::openai_compatible` now re-exports the refreshed Alibaba/Qwen/Mistral model modules so public facade imports match the underlying provider crate.

## Blockers

- None.

## Next Recommended Action

- Do not reopen this lane for DeepInfra. Open a dedicated DeepInfra catalog policy follow-on if full upstream catalog parity is desired.
