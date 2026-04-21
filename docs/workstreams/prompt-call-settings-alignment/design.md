# Prompt Call Settings Alignment - Design

Last updated: 2026-04-21

## Context

`repo-ref/ai/packages/ai/src/prompt/index.ts` still exposes one small but user-visible shared
contract slice that Siumai had not mirrored yet:

- deprecated `CallSettings`
- free timeout helper functions:
  - `getTotalTimeoutMs`
  - `getStepTimeoutMs`
  - `getChunkTimeoutMs`
  - `getToolTimeoutMs`

Siumai already split the real ownership correctly across:

- `LanguageModelCallOptions`
- `RequestOptions`
- `TimeoutConfiguration`

but downstream parity review still had to remember that the compatibility composition layer was
missing.

## Goal

- Add an honest deprecated `CallSettings` compatibility surface on the shared Rust facade.
- Add free timeout helper functions that mirror the AI SDK request-options helpers.
- Keep the ownership model explicit: `CallSettings` is a projection/composition helper, not the
  new primary request contract.

## Non-goals

- Do not re-collapse the runtime back onto `CallSettings`.
- Do not mix this small request/helper slice with the much larger prompt message/content contract.
- Do not pretend `timeout` belongs on deprecated `CallSettings`; AI SDK excludes it there.

## Chosen design

### 1. Keep the primary ownership model unchanged

The stable shared request surface remains:

- `LanguageModelCallOptions`
- `RequestOptions`
- `TimeoutConfiguration`

`CallSettings` is added only as a deprecated compatibility projection above those stable types.

### 2. Model `CallSettings` as a passive Rust struct

Rust does not have TypeScript intersection types, so instead of a fake alias the compatibility
surface is a passive struct containing:

- model-facing generation controls from `LanguageModelCallOptions`
- request-facing non-timeout controls from `RequestOptions`

It intentionally excludes timeout, matching the upstream AI SDK `Omit<RequestOptions, 'timeout'>`
behavior.

### 3. Add explicit projection helpers both ways

`CallSettings` exposes helper projections back onto:

- `language_model_call_options()`
- `request_options()`

and implements `From` conversions from/to those shared types where the projection is lossless.

### 4. Add free timeout helper functions

The shared Rust surface now also exposes:

- `get_total_timeout_ms(...)`
- `get_step_timeout_ms(...)`
- `get_chunk_timeout_ms(...)`
- `get_tool_timeout_ms(...)`

These are thin free-function views over `TimeoutConfiguration`, mirroring the AI SDK helper role
without changing the runtime ownership model.

## Follow-up

The larger `ModelMessage` / `Prompt` / prompt content shared contract remains a separate workstream.
That slice needs its own narrowed shared structs rather than shallow aliases because the current
stable Rust chat model is intentionally richer than the AI SDK prompt-wire contract.
