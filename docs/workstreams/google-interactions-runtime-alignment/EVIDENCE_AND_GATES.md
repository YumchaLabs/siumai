# Google Interactions Runtime Alignment - Evidence And Gates

Status: Active
Last updated: 2026-05-18

## Smallest Current Repro

The Interactions handle now executes non-stream `/interactions` calls and model-mode streaming
`POST /interactions` calls. Agent streaming reconnect/cancel behavior remains the current explicit
deferred boundary:

```bash
cargo nextest run -p siumai-provider-gemini --all-features interactions_streaming_runtime_is_explicitly_deferred --no-fail-fast
cargo nextest run -p siumai --features google google_interactions_package_surface_is_explicitly_deferred_from_chat_runtime --test provider_public_path_parity_test --no-fail-fast
```

## Gate Set

### Request Conversion Gate

```bash
cargo nextest run -p siumai-provider-gemini --all-features google_interactions_request
cargo nextest run -p siumai-provider-gemini --all-features google_interactions_agent
```

Proves stable Siumai request structures produce the Interactions wire body.

### Response Runtime Gate

```bash
cargo nextest run -p siumai-provider-gemini --all-features google_interactions_response
cargo nextest run -p siumai-provider-gemini --all-features google_interactions_non_stream
```

Proves response parsing and non-stream execution use `/interactions`.

### Streaming Runtime Gate

```bash
cargo nextest run -p siumai-provider-gemini --all-features google_interactions_stream
cargo nextest run -p siumai-provider-gemini --all-features google_interactions_stream_reconnect
```

Proves Interactions SSE conversion, reconnect, and cancel behavior.

### Public Path Gate

```bash
cargo nextest run -p siumai --features google google_interactions --test provider_public_path_parity_test --no-fail-fast
```

Proves facade-level behavior matches the provider-owned runtime state.

### Package Gate

```bash
cargo nextest run -p siumai-provider-gemini --all-features --no-fail-fast
```

Proves ordinary Gemini package behavior remains intact.

## Evidence Anchors

- `docs/workstreams/google-interactions-runtime-alignment/DESIGN.md`
- `docs/workstreams/google-interactions-runtime-alignment/TODO.md`
- `docs/workstreams/google-interactions-runtime-alignment/MILESTONES.md`
- `siumai-provider-gemini/src/providers/gemini/interactions.rs`
- `siumai-provider-gemini/src/provider_options/gemini/mod.rs`
- `siumai/tests/provider_public_path_parity_test.rs`
- `repo-ref/ai/packages/google/src/interactions/*`

## Command Log

| Date | Command | Result | Notes |
| --- | --- | --- | --- |
| 2026-05-18 | `git status --short --branch` | Passed | Workstream opened from a clean post-commit tree except the new workstream files. |
| 2026-05-18 | `cargo nextest run -p siumai-provider-gemini --all-features interactions_handle_is_explicitly_deferred_at_runtime --no-fail-fast` | Passed | Focused provider gate passed, proving the current runtime behavior is an explicit fail-fast boundary. |
| 2026-05-18 | `cargo nextest run -p siumai --features google google_interactions_package_surface_is_explicitly_deferred_from_chat_runtime --test provider_public_path_parity_test --no-fail-fast` | Passed | Focused facade gate passed, proving the package-visible Interactions surface is deferred from ordinary chat runtime; existing Gemini unreachable-pattern warning is unrelated. |
| 2026-05-18 | `cargo fmt -p siumai-provider-gemini -- --check` | Passed | Formatting gate passed after splitting Interactions request conversion into a provider-owned submodule. |
| 2026-05-18 | `cargo nextest run -p siumai-provider-gemini --all-features google_interactions_request --no-fail-fast` | Passed | GIR-020 request conversion gate passed: 11 model-mode request tests cover system text, response format, media, tools/tool choice, tool calls/results, signatures, compaction, deprecated `imageConfig`, and warning cases. |
| 2026-05-18 | `cargo nextest run -p siumai-provider-gemini --all-features --no-fail-fast` | Passed | Package gate passed with 88 tests after GIR-020, proving ordinary Gemini provider behavior still passes alongside the new Interactions request conversion tests. |
| 2026-05-18 | `git diff --check` | Passed | Whitespace gate passed for GIR-020 code and workstream documentation updates. |
| 2026-05-18 | `cargo nextest run -p siumai-provider-gemini --all-features google_interactions_agent --no-fail-fast` | Passed | GIR-030 agent request gate passed: 2 tests cover `agent` + `background: true`, `agent_config`, and warning/drop behavior for model-only fields. |
| 2026-05-18 | `cargo nextest run -p siumai-provider-gemini --all-features google_interactions_request --no-fail-fast` | Passed | GIR-020 model request gate still passed after adding agent-mode request conversion: 10 model-mode request tests passed. |
| 2026-05-18 | `cargo nextest run -p siumai-provider-gemini --all-features --no-fail-fast` | Passed | Package gate passed with 89 tests after GIR-030. |
| 2026-05-18 | `git diff --check` | Passed | Whitespace gate passed for GIR-030 code and workstream documentation updates. |
| 2026-05-18 | `cargo nextest run -p siumai-provider-gemini --all-features google_interactions_response --no-fail-fast` | Passed | GIR-040 response parsing gate passed: completed Interactions responses now map to stable `ChatResponse` content, usage, finish reason, provider metadata, sources, images, and tool calls/results. |
| 2026-05-18 | `cargo fmt -p siumai-provider-gemini -- --check` | Passed | Formatting gate passed after adding the provider-owned Interactions response parser module. |
| 2026-05-18 | `cargo nextest run -p siumai-provider-gemini --all-features --no-fail-fast` | Passed | Package gate passed with 92 tests after GIR-040, proving ordinary Gemini provider behavior still passes alongside the new response parser. |
| 2026-05-18 | `cargo clippy -p siumai-provider-gemini --all-features --all-targets -- -D warnings` | Passed | Clippy passed after narrowing dead-code allowances to the deferred Interactions runtime boundary and silencing module inception for `interactions.rs`. |
| 2026-05-18 | `git diff --check` | Passed | Whitespace gate passed for GIR-040 code and workstream documentation updates. |
| 2026-05-18 | `cargo nextest run -p siumai-provider-gemini --all-features google_interactions_non_stream --no-fail-fast` | Passed | GIR-050 non-stream runtime gate passed: 4 capture-transport tests cover model POST, agent polling, missing interaction id errors, and timeout behavior. |
| 2026-05-18 | `cargo fmt -p siumai-provider-gemini -- --check` | Passed | Formatting gate passed after wiring the provider-owned Interactions non-stream runtime. |
| 2026-05-18 | `cargo clippy -p siumai-provider-gemini --all-features --all-targets -- -D warnings` | Passed | Clippy passed for the Gemini provider after GIR-050 runtime wiring. |
| 2026-05-18 | `cargo nextest run -p siumai-provider-gemini --all-features --no-fail-fast` | Passed | Package gate passed with 96 tests after GIR-050, proving ordinary Gemini provider behavior still passes alongside non-stream Interactions runtime. |
| 2026-05-18 | `cargo nextest run -p siumai-provider-gemini --all-features google_interactions_stream --no-fail-fast` | Passed | GIR-060 stream gate passed: 2 capture-transport tests cover model stream POST, text events, reasoning signatures, function-call argument deltas, built-in tool results, sources, image file parts, usage, finish metadata, service tier, and request/response envelopes. |
| 2026-05-18 | `cargo fmt -p siumai-provider-gemini -- --check` | Passed | Formatting gate passed after adding provider-owned Interactions stream conversion. |
| 2026-05-18 | `cargo clippy -p siumai-provider-gemini --all-features --all-targets -- -D warnings` | Passed | Clippy passed for the Gemini provider after GIR-060 stream conversion. |
| 2026-05-18 | `cargo nextest run -p siumai-provider-gemini --all-features --no-fail-fast` | Passed | Package gate passed with 98 tests after GIR-060, proving ordinary Gemini provider behavior remains intact alongside model-mode Interactions streaming. |
