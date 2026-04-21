# Provider Utils Tooling Runtime Alignment - TODO

Last updated: 2026-04-21

Status legend:

- `[ ]` not started
- `[~]` in progress
- `[x]` done
- `[-]` intentionally deferred

## Track A - Shared execution semantics

- [x] Move normalized tool execution results into `siumai-core::tooling`.
- [x] Add `ToolExecutionOptions` plus a shared `ToolExecutionStream`.
- [x] Keep legacy one-shot execution compatible while adding streamed normalization.

## Track B - Tooling facade parity

- [x] Expose `ToolSet`, `tool(...)`, `dynamic_tool(...)`, `is_executable_tool(...)`, and
  `execute_tool(...)`.
- [x] Add public compile/run coverage for shared tooling runtime carrier types on the facade.
- [x] Audit whether additional helper names from `provider-utils` are still missing after the
  current facade uplift.

## Track C - Extras/orchestrator convergence

- [x] Remove the extras-owned duplicate `ToolExecutionResult`.
- [x] Route `ExecutableTools` resolver execution through shared tooling helpers.
- [x] Thread non-empty shared `messages` into `ToolExecutionOptions` where the orchestrator has the
  exact pre-tool-call step input available.
- [x] Unify runtime input callbacks onto shared `ModelMessage` / `Context` / `abort_signal`
  semantics derived from `ToolExecutionOptions`.
- [x] Split approval checks onto a dedicated shared context without `abort_signal`, matching
  upstream `needsApproval(...)` more closely.
- [x] Reuse current continuation message history for resumed approved local tool calls, matching
  AI SDK approval-continuation behavior instead of falling back to empty runtime messages.
- [-] Add a stricter Rust-only pre-tool-call message-slice helper only if it proves valuable beyond
  the AI SDK-aligned continuation-history behavior.

## Track D - Stable tool schema metadata

- [x] Expose `title` on stable tool schemas.
- [x] Expose builder/accessor support for `inputExamples`.
- [x] Expose builder/accessor support for `strict`.
- [x] Expose builder/accessor support for function-tool `providerOptions`.
- [x] Expose builder/accessor support for provider-defined-tool `providerOptions`.

## Track E - Docs and changelog

- [x] Create a dedicated `docs/workstreams/provider-utils-tooling-runtime-alignment/` folder.
- [x] Record the tooling/runtime alignment slice in `CHANGELOG.md` `Unreleased`.
- [x] Record the deliberate no-mirror decision for TS-only `InferTool*` helper types.
