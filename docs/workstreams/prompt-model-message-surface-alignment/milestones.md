# Prompt Model Message Surface Alignment - Milestones

Last updated: 2026-04-21

## Completed

- Audited the upstream AI SDK prompt/message shared contract in `repo-ref/ai` and separated it
  from Siumai's richer `ChatMessage` / `ContentPart` runtime surface.
- Added dedicated prompt-owned shared structs for AI SDK-style model messages, prompt content
  parts, `Prompt`, and `StandardizedPrompt`.
- Added explicit narrowing conversions between `ChatMessage` / `ChatRequest` and the new prompt
  contract, with dedicated validation and conversion error types.
- Re-exported the prompt/message shared surface from `siumai::types::*` and
  `siumai::prelude::unified::*`, and covered the public paths with compile guards plus focused unit
  tests.

## Next

- Continue checking the new prompt/message structs against `repo-ref/ai` for any remaining content
  part or helper gaps.
- Reuse this narrowed shared layer as the baseline when auditing future UI-message and
  provider-utils compatibility slices.
