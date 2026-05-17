# AI SDK Provider Interface Convergence - Handoff

Status: Active
Last updated: 2026-05-18

## Current State

The program workstream is open. The target seams, initial parity inventory, milestones, gates, and
task ledger are recorded. AIPC-030 is complete: `siumai-core::standards` now has an integration
boundary guard against provider/protocol island directories or modules, and the existing registry
family-handle guard was reconfirmed.

This lane is intentionally a coordination and execution program. It should keep spawning bounded
vertical slices instead of becoming one cross-provider mega patch.

## Active Task

- Task ID: AIPC-040
- Owner: codex
- Files:
  - `siumai-registry/src/registry/entry`
  - `docs/workstreams/ai-sdk-provider-interface-convergence/*`
- Validation:
  - focused registry handle tests
  - `cargo nextest run -p siumai-registry --no-fail-fast`

## Decisions Since Last Update

- Opened a new program workstream instead of reopening `fearless-refactor-v4`.
- Kept `ai-sdk-structural-alignment` as historical evidence rather than the active machine-readable
  lane.
- Chose AIPC-030 as the first executable slice because source guards are safer before broader
  provider or registry rewrites.
- AIPC-030 found a stale empty local `siumai-core/src/standards/openai` directory; it was not tracked
  by Git and was removed so the new guard reflects the intended tracked source tree.

## Blockers

- None currently.

## Next Recommended Action

Audit `siumai-registry/src/registry/entry/handles` and factories for remaining primary stable-family
execution that can still route through compatibility clients. If the remaining usages are already
extension-only or compatibility-only, close AIPC-040 with evidence instead of forcing a code move.
