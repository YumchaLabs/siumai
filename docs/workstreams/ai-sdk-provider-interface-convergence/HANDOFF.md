# AI SDK Provider Interface Convergence - Handoff

Status: Active
Last updated: 2026-05-18

## Current State

The program workstream is open. The target seams, initial parity inventory, milestones, gates, and
task ledger are recorded.

This lane is intentionally a coordination and execution program. It should keep spawning bounded
vertical slices instead of becoming one cross-provider mega patch.

## Active Task

- Task ID: AIPC-030
- Owner: codex
- Files:
  - `siumai-core`
  - `siumai-registry`
  - `docs/workstreams/ai-sdk-provider-interface-convergence/*`
- Validation:
  - `cargo nextest run -p siumai-core --no-fail-fast`
  - `cargo nextest run -p siumai-registry --no-fail-fast`

## Decisions Since Last Update

- Opened a new program workstream instead of reopening `fearless-refactor-v4`.
- Kept `ai-sdk-structural-alignment` as historical evidence rather than the active machine-readable
  lane.
- Chose AIPC-030 as the first executable slice because source guards are safer before broader
  provider or registry rewrites.

## Blockers

- None currently.

## Next Recommended Action

Audit existing core and registry boundary tests. Strengthen guards only where they prevent concrete
regressions; if a guard exposes real drift, split the code change into the smallest executable
slice.
