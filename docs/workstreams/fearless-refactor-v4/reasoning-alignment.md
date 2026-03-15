# Fearless Refactor V4 - Reasoning Alignment

Last updated: 2026-03-13

This document tracks the V4 alignment work for **reasoning / thinking** across providers.

It keeps Siumai's Rust-first public naming while borrowing the useful high-level shape from
the Vercel AI SDK: a small Stable contract, provider-owned hint knobs, and predictable
response-side extraction semantics.

## Related docs

- `provider-feature-matrix.md`
- `typed-metadata-boundary-matrix.md`
- `structured-output-parity.md`

## Stable surface (current)

Current Stable reasoning contract is intentionally narrow:

- Request-side defaults:
  - builder / registry default reasoning enable flag
  - builder / registry default reasoning budget
- Response-side extraction:
  - non-streaming: `ChatResponse::reasoning()`
  - streaming: `ChatStreamEvent::ThinkingDelta`

Current non-goals for the Stable surface:

- Do **not** force every provider into a universal `enable + budget` contract.
- Do **not** promote provider-specific effort / format / summary knobs into Stable.
- Do **not** assume `StreamEnd.response.reasoning()` is always the canonical streaming result.

## Canonical public semantics

### Non-streaming

If a provider returns explicit reasoning / thinking text, the Stable response surface should expose it through:

- `ChatResponse::reasoning()`

### Streaming

If a provider emits incremental reasoning / thinking text, the Stable streaming surface should expose it through:

- `ChatStreamEvent::ThinkingDelta`

Public parity should prefer `ThinkingDelta` accumulation over `StreamEnd.response.reasoning()` because:

- some provider / compat streaming paths still finalize text content and usage on `StreamEnd`
  without also replaying accumulated reasoning into the final response object;
- `ThinkingDelta` is the more stable event-level contract for cross-provider convergence today.

### Usage-side counters

Reasoning token counters are useful, but they are **not** the same thing as reasoning text.

- `usage.reasoning_tokens` or provider-owned typed metadata may exist even when no reasoning text
  is surfaced.
- Those counters should be treated as metadata / cost telemetry, not as a substitute for
  `ChatResponse::reasoning()` or `ThinkingDelta`.

## Provider status (audited public stories)

| Provider | Stable request defaults | Provider-owned hint knobs | Non-stream reasoning text | Streaming reasoning text | Current note |
| --- | --- | --- | --- | --- | --- |
| DeepSeek | Yes | `DeepSeekOptions` | `ChatResponse::reasoning()` parity locked | `ThinkingDelta` parity locked | Stable `enable_reasoning` / `reasoning_budget` plus provider-specific normalization are both covered. |
| xAI | Yes | `XaiOptions.reasoningEffort` | `ChatResponse::reasoning()` parity locked | `ThinkingDelta` parity locked | Stable defaults and xAI-specific effort knob are both covered. |
| OpenRouter | Yes (compat vendor view) | `OpenRouterOptions` + compat reasoning defaults | `ChatResponse::reasoning()` parity locked | `ThinkingDelta` parity locked | Compat vendor-view path now has the same response-side parity as native wrappers. |
| Groq | No | `GroqOptions.reasoning_effort` / `reasoning_format` | `ChatResponse::reasoning()` parity locked when provider returns thinking text | `ThinkingDelta` parity locked when provider returns thinking deltas | Groq keeps reasoning hints provider-owned; Stable `enable + budget` is intentionally not claimed. |
| Perplexity | No stable reasoning-text contract | hosted-search typed metadata only | Not currently promoted | Not currently promoted | Current typed boundary is metadata-side only: `usage.reasoning_tokens`, citations, images, search counters. |

## Public-path parity rule

For any provider enrolled in reasoning alignment, builder / provider / config-first / registry
construction should agree on:

1. the final reasoning-related request shape that provider is supposed to support;
2. the non-streaming reasoning extraction result, if the provider returned reasoning text;
3. the streaming `ThinkingDelta` accumulation semantics, if the provider streamed reasoning text.

That rule now explicitly covers:

- DeepSeek
- xAI
- OpenRouter
- Groq

## Boundary rules

### Keep Stable minimal

Use Stable only for:

- default reasoning enable / budget where the provider contract is genuinely shared;
- generic reasoning text extraction (`ChatResponse::reasoning()`, `ThinkingDelta`).

### Keep provider hints provider-owned

Keep these under provider-owned typed options:

- xAI `reasoningEffort`
- Groq `reasoning_effort`
- Groq `reasoning_format`
- any future provider-specific summary / verbosity / search-coupled reasoning knob

### Keep metadata separate from reasoning text

Do not widen Stable response semantics from usage-side counters alone.

Examples:

- Perplexity `usage.reasoning_tokens` stays typed metadata, not Stable reasoning text.
- provider-specific cost / search / citation fields stay metadata even when they correlate with reasoning.

## Recommended test strategy

When advancing reasoning parity for a provider:

1. lock request shaping first;
2. lock non-stream `ChatResponse::reasoning()` if the provider returns reasoning text;
3. lock streaming `ThinkingDelta` accumulation if the provider streams reasoning text;
4. only then decide whether `StreamEnd.response.reasoning()` is a guaranteed public contract.

## Known gaps / next steps

- Anthropic / Gemini reasoning response-side parity is not yet summarized here at the same public-path level.
- If a provider eventually guarantees accumulated reasoning on `StreamEnd.response`, we can widen
  the public streaming contract and update this document in the same change.
