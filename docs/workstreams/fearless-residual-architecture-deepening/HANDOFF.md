# Fearless Residual Architecture Deepening - Handoff

Status: Active
Last updated: 2026-05-27

## Current State

The workstream is open for five residual architecture-review candidates. FRAD-010 and FRAD-020 are
complete. The next executable task is FRAD-030, the OpenAI-compatible message dialect conversion
seam.

## Active Task

- Task ID: FRAD-030
- Owner: codex
- Files:
  - `siumai-protocol-openai/src/standards/openai`
  - `CHANGELOG.md`
  - `siumai-protocol-openai/CHANGELOG.md`
- Validation:
  - `cargo fmt --check -p siumai-protocol-openai`
  - `cargo nextest run -p siumai-protocol-openai --features openai-standard,openai-responses openai --no-fail-fast`
- Status: READY
- Review: not started
- Evidence: `EVIDENCE_AND_GATES.md`

## Decisions

- Use one durable workstream for all five candidates because they share the same post-AI-SDK
  residual architecture review and closeout gate.
- Do not open a new ADR yet. Existing ADRs cover the direction; `ContentPart` remains controlled by
  ADR-0008.
- Execute in dependency order: registry descriptor first, protocol dialects second, bridge codecs
  third, test harness fourth, `ContentPart` gate/move fifth.
- FRAD-020 concentrated built-in provider default-model lookup, factory resolution, and enabled
  factory registration in `registry::provider_descriptor`; the public helper functions now cross
  that seam, and catalog projection uses `ProviderCatalogDescriptor`.

## Next Recommended Action

Commit FRAD-020, then run FRAD-030 with the OpenAI protocol crate as the bounded scope.
