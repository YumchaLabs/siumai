# Provider Splitting Plan (Vercel AI SDK-aligned)

This document proposes an incremental, low-risk refactor plan to reduce coupling across providers
in `siumai`, while keeping fixture-driven behavior stable.

## Goals

- Align package granularity with Vercel AI SDK provider packages:
  - Each provider owns its wire-level quirks (naming, stream parts shape, tool mapping edge-cases).
  - Shared "protocol" code stays vendor-agnostic.
- Reduce cross-provider switches in shared code (`if xai then ...`).
- Keep public API stable (or evolve behind new, additive APIs).
- Make fixture alignment the source of truth.

## Current Situation (high-level)

- We already have split crates:
  - `siumai-core` (types, execution, streaming primitives)
  - `siumai-protocol-*` protocol crates (some are still thin compatibility re-exports during the migration)
  - `siumai-provider-*` provider crates (OpenAI, Anthropic, Gemini, xAI, etc.)
- However, xAI behavior currently requires "mode" toggles in shared OpenAI Responses logic:
  - SSE stream parts shape (`reasoning-*` / `text-*`, finish reason raw, usage omissions)
  - Tool stream behavior (`web_search`/`x_search` input parts)
  - doGenerate response mapping (`code_execution`, `web_search`, `x_search`)

This is correct functionally, but it increases coupling because shared protocol code must know
about vendor-specific semantics.

## Target Architecture

### 1) Protocol crates (vendor-agnostic)

- `siumai-protocol-openai`
  - OpenAI ChatCompletions + Responses protocol mapping
  - Generic tools conversion utilities
  - Generic SSE converter for Responses
- `siumai-protocol-anthropic`
- `siumai-protocol-gemini`

Protocol crates should avoid referencing provider names directly (e.g. `xai`, `deepseek`).

### 2) Provider crates (vendor-specific)

Each provider crate owns:

- Default configuration and presets (e.g. stream parts shape)
- Any mapping differences not representable in the protocol baseline
- Provider-specific fixtures + alignment tests

Examples:

- `siumai-provider-xai`
  - Exposes `XaiResponsesEventConverter` preset instead of requiring callers to set
    `StreamPartsStyle::Xai` / `WebSearchStreamMode::Xai` manually.
  - Exposes `XaiResponsesResponseTransformer` (optional) if we want to remove heuristics
    from shared transformers.

## Incremental Refactor Steps

### Step A: Add provider-owned presets (low risk, additive)

- Add provider wrapper types that preconfigure shared protocol converters/transformers.
- Update provider tests and examples to use these presets.
- Keep the underlying "mode" hooks for now; mark them as protocol-level extension points.

### Step B: Move vendor differences behind provider wrappers

- Replace heuristics (e.g. `is_xai_response` detection) with explicit provider-owned transformers
  where feasible.
- Narrow the protocol API to the minimal knobs required.

### Step C: Optional crate-level extraction

- If vendor logic grows, introduce `siumai-protocol-xai` as a thin wrapper over
  `siumai-protocol-openai` plus xAI-specific behavior.
- Keep `siumai-provider-xai` depending on `siumai-protocol-xai`.

## Non-goals (for now)

- Renaming crates / breaking API surface in `siumai` root crate.
- Removing all compatibility re-exports in one shot.
- Large-scale module moves without fixture coverage.

## Acceptance Criteria

- All existing alignment tests pass.
- New provider presets are used by xAI tests (no manual toggles).
- Shared protocol code remains stable and smaller in responsibility.
