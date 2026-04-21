# Provider Utils Tooling Runtime Alignment - Milestones

Last updated: 2026-04-21

## M1 - Shared execution uplift

Status: in progress

- move normalized `preliminary` / `final` tool execution results into `siumai-core::tooling`
- keep backward compatibility for one-shot `ExecutableTool` executors
- add shared execution options and normalized execution stream helpers

## M2 - AI SDK-style tooling facade

Status: in progress

- expose `ToolSet`, `tool(...)`, `dynamic_tool(...)`, `is_executable_tool(...)`,
  and `execute_tool(...)`
- add public compile guards for the new tooling surface

## M3 - Extras integration

Status: in progress

- remove extras-owned duplicate `ToolExecutionResult`
- route `ExecutableTools` resolver execution through shared tooling stream helpers

## M4 - Stable schema builder completion

Status: completed

- expose `title`, `inputExamples`, `strict`, and function-tool `providerOptions` builders/accessors
- expose provider-defined-tool `providerOptions` on the stable portable tool surface
- add stable schema roundtrip coverage

## M5 - Follow-up audit

Status: in progress

- unify runtime callback contexts on top of the shared execution-options contract
- split approval checks onto a dedicated shared context that mirrors upstream `needsApproval(...)`
- thread stream cancel handles into tool runtime callbacks and local tool execution
- reuse current continuation message history for approved local-tool resumes during approval
  preprocess

## M6 - Remaining audit gaps

Status: deferred

- evaluate remaining `provider-utils` infer/helper surfaces
- decide whether a Rust-only stricter pre-tool-call message-slice helper is worth adding beyond
  the AI SDK-aligned continuation-history behavior
