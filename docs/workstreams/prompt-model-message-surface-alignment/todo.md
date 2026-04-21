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
- [x] Keep prompt-owned `ToolApprovalResponse` optional fields ergonomic through builder helpers
  and public facade coverage.
- [x] Keep prompt-owned `providerOptions` fields ergonomic through shared builder/accessor helpers
  on prompt parts and model-message structs.
- [x] Match the wider shared-type convention by exposing single-provider prompt helpers
  `with_provider_option(...)` / `provider_option(...)`.
- [x] Match the wider shared-type convention on `ToolResultOutput` and
  `ToolResultContentPart` with `provider_options_map*`, `with_provider_options_map(...)`, and
  `provider_option(...)` helpers.
- [x] Add field-level prompt-part builders for optional `ImagePart.mediaType` and
  `FilePart.filename`.

## Track C - Docs and changelog

- [x] Create a dedicated `docs/workstreams/prompt-model-message-surface-alignment/` folder.
- [x] Record the new shared prompt/message surface in `CHANGELOG.md` `Unreleased`.

## Track D - Intentional boundaries

- [-] Do not alias the prompt/message surface directly onto `ChatMessage` / `ContentPart`.
- [-] Do not silently drop unsupported runtime/provider fields during narrowing conversion.
