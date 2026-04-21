# Prompt Model Message Surface Alignment - TODO

Last updated: 2026-04-21

Status legend:

- `[ ]` not started
- `[~]` in progress
- `[x]` done
- `[-]` intentionally deferred

## Track A - Shared prompt contract

- [x] Audit the AI SDK prompt/message shared surface in `repo-ref/ai`.
- [x] Add dedicated prompt-owned Rust structs for `ModelMessage`, `Prompt`, and prompt content
  parts.
- [x] Keep the shared prompt contract narrower than `ChatMessage` / `ContentPart`.
- [x] Add explicit prompt validation and narrowing conversion errors.
- [x] Require exact prompt `role` / `type` discriminators during serde deserialization.

## Track B - Facade and tests

- [x] Re-export the prompt/message shared surface from `siumai-spec`, `siumai::types::*`, and
  `siumai::prelude::unified::*`.
- [x] Add public compile-guard coverage for the new prompt/message types and conversion entry
  points.
- [x] Add focused unit coverage for prompt standardization and conversion failures.

## Track C - Docs and changelog

- [x] Create a dedicated `docs/workstreams/prompt-model-message-surface-alignment/` folder.
- [x] Record the new shared prompt/message surface in `CHANGELOG.md` `Unreleased`.

## Track D - Intentional boundaries

- [-] Do not alias the prompt/message surface directly onto `ChatMessage` / `ContentPart`.
- [-] Do not silently drop unsupported runtime/provider fields during narrowing conversion.
