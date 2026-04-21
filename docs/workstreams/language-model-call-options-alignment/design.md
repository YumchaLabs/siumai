# Language Model Call Options Alignment - Design

Last updated: 2026-04-21

## Problem

Compared with `repo-ref/ai/packages/ai/src/prompt/language-model-call-options.ts`, Siumai already
had most model-facing generation controls in `CommonParams`, but the public shared surface still
drifted in two important ways:

- there was no dedicated shared `LanguageModelCallOptions` view on the stable Rust facade
- `CommonParams` itself had a structural hole: `CommonParamsBuilder` could not set
  `max_completion_tokens`, and `cache_hash()` ignored that field even though it changes request
  semantics

This made shared contract audits noisy and also left a real correctness risk in any cache or
builder path that depends on `max_completion_tokens`.

## Goals

- Expose an honest AI SDK-style `LanguageModelCallOptions` projection on the stable Rust surface.
- Keep this workstream limited to model-facing generation controls.
- Fix the `CommonParams` structural gaps that block accurate projection.

## Non-goals

- Do not widen this workstream into request-facing transport controls such as
  `RequestOptions` or `TimeoutConfiguration`.
- Do not pretend Siumai already has a stable cross-provider request lane for `reasoning`.

## Chosen design

### 1. Add a dedicated shared projection type

`siumai-spec/src/types/ai_sdk.rs` now exposes:

- `LanguageModelCallOptions`
- `LanguageModelReasoning`

This type mirrors the shared AI package contract shape:

- `maxOutputTokens`
- `temperature`
- `topP`
- `topK`
- `presencePenalty`
- `frequencyPenalty`
- `stopSequences`
- `seed`
- `reasoning`

### 2. Treat it as a projection from `CommonParams`

The current stable runtime already stores most of these controls in `CommonParams`, so
`LanguageModelCallOptions` is implemented as an honest projection:

- `From<&CommonParams>` and `From<CommonParams>` exist
- `max_output_tokens` prefers `max_completion_tokens` and falls back to `max_tokens`
- `reasoning` currently stays `None` because Siumai still lacks a stable cross-provider request
  lane for it

This keeps the public structure useful without claiming runtime support that does not exist yet.

### 3. Fix the real `CommonParams` defects instead of layering over them

This workstream also fixes two underlying problems:

- `CommonParamsBuilder` now supports `max_completion_tokens`
- `CommonParams::cache_hash()` now includes `max_completion_tokens`

Without those fixes, the new shared projection would still rest on an incomplete local contract.

## Validation

This slice is locked by:

- `siumai-spec/src/types/{ai_sdk,params}.rs` unit tests
- public compile guards in `siumai/tests/public_surface_imports_test.rs`
- targeted compile/test commands for `siumai-spec` and `siumai`

## Deferred follow-up

- design `RequestOptions` and `TimeoutConfiguration` as a separate workstream
- revisit whether stable cross-provider `reasoning` should live in `CommonParams`, a dedicated
  request-layer helper, or remain provider-owned
