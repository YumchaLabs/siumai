# AI SDK Provider Interface Convergence - Handoff

Status: Active
Last updated: 2026-05-18

## Current State

The program workstream is open. The target seams, initial parity inventory, milestones, gates, and
task ledger are recorded. AIPC-030 and AIPC-040 are complete. AIPC-050 is in progress with the
first OpenAI Responses stream-part slice landed locally: public feature-surface tests now exercise
stable `ChatStreamEvent::Part` tool call/result inputs instead of provider custom event inputs, while
the converter-level custom-event compatibility tests remain in place.

This lane is intentionally a coordination and execution program. It should keep spawning bounded
vertical slices instead of becoming one cross-provider mega patch.

## Active Task

- Task ID: AIPC-050
- Owner: codex
- Files:
  - `siumai-core`
  - `siumai-protocol-*`
  - `siumai-extras`
  - `docs/workstreams/ai-sdk-provider-interface-convergence/*`
- Validation:
  - focused protocol stream tests for the affected provider
  - package-scoped `cargo nextest`

## Decisions Since Last Update

- Opened a new program workstream instead of reopening `fearless-refactor-v4`.
- Kept `ai-sdk-structural-alignment` as historical evidence rather than the active machine-readable
  lane.
- Chose AIPC-030 as the first executable slice because source guards are safer before broader
  provider or registry rewrites.
- AIPC-030 found a stale empty local `siumai-core/src/standards/openai` directory; it was not tracked
  by Git and was removed so the new guard reflects the intended tracked source tree.
- AIPC-040 found that remaining registry handle compatibility clients are already extension-only:
  image edit/variation helpers and audio streaming/listing/translation helpers. A new boundary test
  locks that shape so primary stable-family execution does not regress.
- AIPC-050 started with OpenAI Responses because its public feature surface still modeled stable
  tool stream parts through `Custom("openai:*")` inputs even though production serialization already
  accepts `ChatStreamEvent::Part`.

## Blockers

- None currently.

## Next Recommended Action

Continue AIPC-050 by auditing Anthropic/Gemini and `siumai-extras` gateway stream tests for any
remaining stable AI SDK semantics asserted through provider custom/raw replay data. Keep legacy
custom-event compatibility tests only where they explicitly prove backward compatibility.
