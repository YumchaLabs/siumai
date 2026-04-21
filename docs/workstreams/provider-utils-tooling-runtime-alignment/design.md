# Provider Utils Tooling Runtime Alignment - Design

Last updated: 2026-04-21

## Problem

After the recent shared prompt/message/type alignment work, Siumai still drifted from
`repo-ref/ai/packages/provider-utils/src/types/*` in one especially important area:
tooling/runtime helpers.

The stable Rust surface already had:

- portable `Tool` / `ProviderDefinedTool`
- runtime `ExecutableTool` / `ExecutableTools`
- AI SDK-style runtime metadata for approval, dynamic tools, input callbacks, and model-output
  mapping

But the execution layer was still incomplete relative to the upstream helper contract:

- shared tooling only supported one-shot `execute_json(...) -> Value`
- streamed preliminary/final tool execution lived in `siumai-extras::orchestrator`, not in the
  shared tooling layer
- there was no AI SDK-style `ToolSet`, `tool(...)`, `dynamic_tool(...)`,
  `is_executable_tool(...)`, or `execute_tool(...)` helper surface
- stable tool schema fields such as `title`, `inputExamples`, `strict`, and function-tool
  `providerOptions` did not yet have a complete builder/accessor surface on the Rust side

That split made audits against `provider-utils` noisy and forced higher-level code to depend on
extras-specific execution semantics for functionality that should live in the shared tooling layer.

## Goals

- Move normalized preliminary/final tool execution semantics into `siumai-core::tooling`.
- Expose AI SDK-style helper names on the stable Rust tooling surface.
- Keep backward compatibility for existing one-shot Rust tool executors.
- Make `siumai-extras` orchestrator reuse shared tooling execution instead of owning a parallel
  execution-result type.
- Fill the most obvious stable tool-schema builder/accessor gaps that block mechanical comparison
  against upstream.

## Non-goals

- Do not force the entire orchestrator/runtime pipeline to use `ModelMessage` internally in this
  slice.
- Do not pretend Siumai already threads full AI SDK execution options through every higher-level
  caller; some adapters still only have partial context available.
- Do not introduce TypeScript-style generic infer helpers mechanically when Rust trait/object
  design would make them dishonest or low-value.

## Chosen design

### 1. Shared tooling now owns normalized tool execution results

`siumai-core::tooling` now defines the shared execution carrier:

- `ToolExecutionOptions`
- `ToolExecutionResult`
- `ToolExecutionStream`

This moves streamed tool execution out of `siumai-extras`-only ownership and makes the capability
available at the same layer as `ExecutableTool`.

### 2. Keep backward compatibility by supporting three execution lanes

`ExecutableTool` now supports three execution bindings:

- legacy one-shot `with_execute(...)`
- one-shot `with_execute_with_options_fn(...)`
- raw streamed `with_execute_stream_fn(...)`

The shared `execute_tool(...)` helper normalizes them into one AI SDK-style stream:

- one-shot executors emit a single `final` result
- stream executors emit every raw value as `preliminary` and replay the last one as `final`

This mirrors the upstream `executeTool(...)` behavior closely without breaking existing Rust code.

### 3. Add AI SDK-style helper names on the stable tooling surface

The shared tooling layer now also exposes:

- `ToolSet`
- `tool(...)`
- `dynamic_tool(...)`
- `is_executable_tool(...)`
- `execute_tool(...)`

These are compatibility-oriented helpers over the existing Rust carriers, not a second parallel
tool system.

### 4. Rewire extras to consume the shared execution layer

`siumai-extras::orchestrator::ToolResolver` now reuses the shared `ToolExecutionResult` type and
`ExecutableTools` resolves streaming execution through shared tooling APIs instead of its own local
execution-result enum.

This reduces duplication and makes future audits against `provider-utils` more mechanical.

### 5. Finish the obvious stable tool-schema builder surface

Stable tool definitions now expose first-class builders/accessors for:

- `title`
- `inputExamples`
- `strict`
- function-tool `providerOptions`

These metadata fields were already present or clearly intended by the AI SDK-aligned schema design,
but the Rust builder surface lagged behind.

## Validation

This workstream is locked by:

- `cargo check --workspace`
- `cargo nextest run -p siumai-spec`
- `cargo nextest run -p siumai --test public_surface_imports_test`

## Deferred follow-up

- Audit whether `ToolExecutionOptions.messages` and runtime callback contexts should be unified on
  top of `ModelMessage` end-to-end instead of the current mixed shared/runtime representation.
- Revisit whether provider-defined tools should also carry first-class stable `providerOptions`
  metadata on the portable schema surface.
- Evaluate whether a Rust-idiomatic equivalent of upstream `InferToolContext` /
  `InferToolSetContext` adds enough value to justify the additional generic surface.
