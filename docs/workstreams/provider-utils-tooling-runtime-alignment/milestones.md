# Provider Utils Tooling Runtime Alignment - Milestones

Last updated: 2026-04-21

## M1 - Shared execution uplift

Status: completed

- move normalized `preliminary` / `final` tool execution results into `siumai-core::tooling`
- keep backward compatibility for one-shot `ExecutableTool` executors
- add shared execution options and normalized execution stream helpers

## M2 - AI SDK-style tooling facade

Status: completed

- expose `ToolSet`, `tool(...)`, `dynamic_tool(...)`, `is_executable_tool(...)`,
  and `execute_tool(...)`
- add public compile guards for the new tooling surface

## M3 - Extras integration

Status: completed

- remove extras-owned duplicate `ToolExecutionResult`
- route `ExecutableTools` resolver execution through shared tooling stream helpers

## M4 - Stable schema builder completion

Status: completed

- expose `title`, `inputExamples`, `strict`, and function-tool `providerOptions` builders/accessors
- expose provider-defined-tool `providerOptions` on the stable portable tool surface
- add stable schema roundtrip coverage

## M5 - Follow-up audit

Status: completed

- unify runtime callback contexts on top of the shared execution-options contract
- split approval checks onto a dedicated shared context that mirrors upstream `needsApproval(...)`
- thread stream cancel handles into tool runtime callbacks and local tool execution
- reuse current continuation message history for approved local-tool resumes during approval
  preprocess

## M6 - Remaining audit gaps

Status: completed

- evaluate remaining `provider-utils` infer/helper surfaces
- record the deliberate decision not to mirror TS-only `InferTool*` conditional types on the
  stable Rust facade
- add public compile/run coverage for the shared runtime carrier types

## M7 - Deferred ideas

Status: deferred

- decide whether a Rust-only stricter pre-tool-call message-slice helper is worth adding beyond
  the AI SDK-aligned continuation-history behavior
