# Request Options Alignment - Design

Last updated: 2026-04-24

## Problem

Compared with `repo-ref/ai/packages/ai/src/prompt/request-options.ts`, Siumai already had the
practical pieces of request-facing transport control, but they were split across unrelated layers:

- retry lived in facade-level helper option structs through `RetryOptions`
- headers and total timeout lived on per-request `HttpConfig`
- cancellation lived in `siumai-core` streaming helpers through `CancelHandle`
- there was no shared public `RequestOptions` or `TimeoutConfiguration` contract that could be
  audited directly against the AI SDK prompt package

This meant the runtime could often do the right thing, but the stable shared shape was missing.

## Goals

- Expose a stable shared `TimeoutConfiguration` contract.
- Expose a stable shared `RequestOptions` contract.
- Fix the ownership problem around cancellation by moving the reusable cancel handle to
  `siumai-spec` instead of keeping it hidden in `siumai-core`.
- Keep the implementation honest about what is already runtime-wired and what is not.

## Non-goals

- Do not pretend that all `RequestOptions` fields are already consumed uniformly by every helper.
- Do not force `chunkMs`, `stepMs`, or per-tool timeout semantics into runtimes that do not yet
  have matching execution lanes.
- Do not remove the Rust-first `retry` / `timeout` / `headers` fields until downstream callers
  have a clear migration window.

## Chosen design

### 1. Move the reusable cancel handle to the shared type layer

`CancelHandle` is now owned by `siumai-spec` and re-exported through the stable public facade.

That fixes the dependency-direction problem for `RequestOptions.abort_signal`, because the shared
request contract can now reference a stable cancellation handle without depending on `siumai-core`.

### 2. Add shared timeout data structures with helper accessors

The shared Rust surface now has:

- `TimeoutConfiguration`
- `TimeoutConfigurationSettings`

These types mirror the AI SDK request-time timeout concept closely enough to audit directly while
staying honest about Rust ergonomics:

- a plain integer total timeout is modeled as `TimeoutConfiguration::Millis`
- structured settings are modeled explicitly
- per-tool overrides are stored in the AI SDK-style `{toolName}Ms` keyed map
- helper accessors expose total/step/chunk/tool timeout lookups plus `Duration` conversion

### 3. Add a shared `RequestOptions` carrier

The shared Rust surface now also has:

- `RequestOptions`

It carries:

- `max_retries`
- `abort_signal`
- `headers`
- `timeout`

This is intentionally a runtime-facing carrier, not a wire-serialized request payload. It therefore
does not derive serde and includes convenience helpers such as:

- `effective_headers()`
- `max_attempts()`
- `total_timeout()`
- `step_timeout()`
- `chunk_timeout()`
- `tool_timeout_ms(...)`

### 4. Adopt `RequestOptions` at the facade helper layer

All stable family helper option structs now accept `request_options: Option<RequestOptions>`:

- text: `GenerateOptions`, `StreamOptions`
- completion: `CompleteOptions`, `StreamOptions`
- embedding: `EmbedOptions`
- image: `GenerateOptions`
- video: `CreateTaskOptions`, `QueryTaskOptions`, `WaitForTaskOptions`, `GenerateOptions`
- speech: `SynthesizeOptions`
- transcription: `TranscribeOptions`
- rerank: `RerankOptions`

The mapping is intentionally centralized in the `siumai` facade crate:

- `max_retries` maps to policy `RetryOptions` attempts as `maxRetries + 1`
- when `request_options` is present and `max_retries` is omitted, the AI SDK default of 2 retries
  is used
- existing Rust-first `retry` fields override `request_options.max_retries`
- `headers` are materialized by filtering out `None` values and then merged into request
  `HttpConfig.headers`
- existing Rust-first `headers` fields override same-name `request_options.headers`
- `timeout.totalMs` maps to request `HttpConfig.timeout` where the helper owns a request object
- existing Rust-first `timeout` fields override `request_options.timeout.totalMs`
- `abort_signal` cancels non-streaming helper calls, stream handshakes, ordinary stream
  consumption, `stream_with_cancel` handles, and video polling loops

### 5. Keep runtime wiring status explicit

Current status after this slice:

- shared request-facing data structures now exist on the stable facade
- `CancelHandle` is no longer trapped inside `siumai-core`
- stream/runtime internals already reuse the shared cancel handle type
- helper-family-wide consumption of the transport subset is implemented through facade adapters

The remaining limits are explicit:

- `stepMs` is exposed as data and helper accessors, but the Rust helper runtime does not yet own an
  AI SDK-style multi-step generation loop where it can enforce the value
- `chunkMs` is exposed as data and helper accessors, but stream idle-timeout enforcement is still a
  follow-up because it should live in the shared stream lane instead of per-helper wrappers
- `toolMs` and per-tool `{toolName}Ms` overrides are exposed as data and helper accessors, but the
  generic facade does not yet own a stable cross-provider tool execution scheduler
- `video::query_task` has no generic request object, so `headers` and request `HttpConfig.timeout`
  are only honored on task submission; query retry and abort are still honored

## Validation

This slice is currently locked by:

- `cargo check -p siumai-spec --tests`
- `cargo nextest run -p siumai --test public_surface_imports_test`
- `cargo nextest run -p siumai --lib`
- unit coverage in `siumai-spec/src/types/ai_sdk.rs`
- facade adapter coverage in `siumai/src/request_options.rs`
- workspace compilation after moving `CancelHandle` ownership

## Deferred follow-up

- decide where `chunkMs` and `stepMs` should land in runtime stream/execution lanes
- decide where generic tool timeout scheduling should live
- revisit whether `headers: Record<string, string | undefined>` needs a stronger Rust delete/merge
  story than the current filtered optional-map carrier
