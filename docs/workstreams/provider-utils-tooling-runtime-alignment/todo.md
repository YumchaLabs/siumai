# Provider Utils Tooling Runtime Alignment - TODO

Last updated: 2026-04-21

Status legend:

- `[ ]` not started
- `[~]` in progress
- `[x]` done
- `[-]` intentionally deferred

## Track A - Shared execution semantics

- [~] Move normalized tool execution results into `siumai-core::tooling`.
- [~] Add `ToolExecutionOptions` plus a shared `ToolExecutionStream`.
- [~] Keep legacy one-shot execution compatible while adding streamed normalization.

## Track B - Tooling facade parity

- [~] Expose `ToolSet`, `tool(...)`, `dynamic_tool(...)`, `is_executable_tool(...)`, and
  `execute_tool(...)`.
- [ ] Audit whether additional helper names from `provider-utils` are still missing after the
  current facade uplift.

## Track C - Extras/orchestrator convergence

- [~] Remove the extras-owned duplicate `ToolExecutionResult`.
- [~] Route `ExecutableTools` resolver execution through shared tooling helpers.
- [ ] Decide whether higher-level adapters should also thread non-empty shared `messages` into
  `ToolExecutionOptions`.

## Track D - Stable tool schema metadata

- [~] Expose `title` on stable tool schemas.
- [~] Expose builder/accessor support for `inputExamples`.
- [~] Expose builder/accessor support for `strict`.
- [~] Expose builder/accessor support for function-tool `providerOptions`.
- [ ] Audit whether provider-defined tools also need stable `providerOptions` to match upstream
  expectations more closely.

## Track E - Docs and changelog

- [x] Create a dedicated `docs/workstreams/provider-utils-tooling-runtime-alignment/` folder.
- [ ] Record the tooling/runtime alignment slice in `CHANGELOG.md` `Unreleased`.
