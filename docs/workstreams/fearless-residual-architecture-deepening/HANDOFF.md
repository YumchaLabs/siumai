# Fearless Residual Architecture Deepening - Handoff

Status: Active
Last updated: 2026-05-27

## Current State

The workstream is open for five residual architecture-review candidates. FRAD-010 completed planning
and documentation. The first executable code task is FRAD-020, the registry provider descriptor
seam.

## Active Task

- Task ID: FRAD-020
- Owner: codex
- Files:
  - `siumai-registry/src/provider_catalog.rs`
  - `siumai-registry/src/native_provider_metadata.rs`
  - `siumai-registry/src/provider/catalog_ids.rs`
  - `siumai-registry/src/registry/helpers.rs`
  - `siumai-registry/src/registry/mod.rs`
  - `siumai-registry/src/registry/factories`
  - `CHANGELOG.md`
  - `siumai-registry/CHANGELOG.md`
- Validation:
  - `cargo fmt --check -p siumai-registry`
  - `cargo nextest run -p siumai-registry provider_catalog --no-fail-fast`
  - `cargo nextest run -p siumai-registry factory_architecture_boundary_test --no-fail-fast`
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

## Next Recommended Action

Commit FRAD-010, then run FRAD-020 with the registry crate as the bounded scope.
