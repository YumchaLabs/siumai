# Language Model Call Options Alignment - TODO

Last updated: 2026-04-21

Status legend:

- `[ ]` not started
- `[~]` in progress
- `[x]` done
- `[-]` intentionally deferred

## Track A - Shared type surface

- [x] Audit `repo-ref/ai/packages/ai/src/prompt/language-model-call-options.ts`.
- [x] Add `LanguageModelCallOptions` to the stable shared Rust surface.
- [x] Add a dedicated `LanguageModelReasoning` enum for the shared projection.
- [x] Re-export the new types from `siumai::types::*` and `siumai::prelude::unified::*`.

## Track B - Local structural correctness

- [x] Add `CommonParamsBuilder::max_completion_tokens(...)`.
- [x] Include `max_completion_tokens` in `CommonParams::cache_hash()`.
- [x] Add unit coverage for both behaviors.

## Track C - Intentional limitations

- [x] Keep `LanguageModelCallOptions.reasoning` as a passive shared field for now.
- [-] Do not claim a stable cross-provider request wiring for `reasoning` until the request lane
  is designed explicitly.
- [-] Do not bundle `RequestOptions` or `TimeoutConfiguration` into this workstream.
