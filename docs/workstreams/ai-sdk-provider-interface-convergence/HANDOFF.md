# AI SDK Provider Interface Convergence - Handoff

Status: Active
Last updated: 2026-05-18

## Current State

The program workstream is open. The target seams, initial parity inventory, milestones, gates, and
task ledger are recorded. AIPC-030, AIPC-040, and AIPC-050 are complete. AIPC-050 closed after three
stream-part slices: OpenAI Responses public feature-surface tests now exercise stable
`ChatStreamEvent::Part` tool call/result inputs instead of provider custom event inputs; extras
gateway smoke tests now require stable downstream tool stream parts for the Anthropic-to-OpenAI
Responses route; Anthropic and Gemini serializer tests now have custom-input source guards so stable
serializer behavior cannot be covered only through custom event inputs. Converter-level custom-event
compatibility tests remain in place where they explicitly prove backward compatibility or
provider-native replay behavior.

This lane is intentionally a coordination and execution program. It should keep spawning bounded
vertical slices instead of becoming one cross-provider mega patch.

## Active Task

- Task ID: AIPC-060
- Owner: codex
- Files:
  - `siumai-bridge`
  - `siumai-extras`
  - `docs/workstreams/ai-sdk-provider-interface-convergence/*`
- Validation:
  - `cargo nextest run -p siumai-bridge --no-fail-fast`
  - focused `siumai-extras` gateway tests

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
- AIPC-050 then tightened the extras gateway smoke helper from typed-or-custom to stable-part-only
  so gateway tests cannot mask a regression from stable tool parts back to provider custom payloads.
- AIPC-050 added protocol serializer source guards for Anthropic and Gemini: `Custom` serializer
  inputs are allowed only in explicitly named V3 compatibility, provider-native, or compatibility
  tests.
- AIPC-050 intentionally did not remove loose custom-event parsers in extras object/tool-loop/server
  helpers. Those are compatibility-boundary consumers rather than stable protocol feature-surface
  tests and should be audited under AIPC-060 only if bridge/gateway evidence shows they can hide a
  stable-part regression.

## Blockers

- None currently.

## Next Recommended Action

Execute AIPC-060. Start with `siumai-bridge` stream tests and bridge conversion helpers, then audit
the remaining extras gateway/transcode assertions for stable-part-first coverage. Keep Axum/server
transport concerns out of `siumai-core`.
