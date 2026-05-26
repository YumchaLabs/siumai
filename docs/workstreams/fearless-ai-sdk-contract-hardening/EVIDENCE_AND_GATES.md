# Fearless AI SDK Contract Hardening - Evidence And Gates

Status: Active
Last updated: 2026-05-26

## Smallest Current Repro

The initial audit is a static contract mismatch, not a runtime failure. The smallest repro is:

```bash
python .agents/skills/siumai-ai-sdk-maintenance/scripts/resolve_ai_sdk_repo.py
cargo nextest run -p siumai-core streaming::processor --no-fail-fast
```

The first command proves the local Vercel AI SDK reference is available. The second is the first
targeted gate for stream final-response assembly once AICH-020 starts.

## Gate Set

### Workstream Bootstrap Gate

```bash
python -m json.tool docs/workstreams/fearless-ai-sdk-contract-hardening/WORKSTREAM.json
git diff --check -- CHANGELOG.md docs/workstreams/fearless-ai-sdk-contract-hardening docs/workstreams/INDEX.md
```

### Stream Replay Gate

```bash
cargo nextest run -p siumai-spec stream --no-fail-fast
cargo nextest run -p siumai-core streaming::processor --no-fail-fast
cargo nextest run -p siumai-protocol-openai responses_sse --no-fail-fast
```

### Diagnostics And Error Safety Gate

```bash
cargo nextest run -p siumai-spec metadata --no-fail-fast
cargo nextest run -p siumai-core error --no-fail-fast
```

### Tool And Usage Gate

```bash
cargo nextest run -p siumai-spec tools --no-fail-fast
cargo nextest run -p siumai-spec usage --no-fail-fast
cargo nextest run -p siumai-core streaming::processor --no-fail-fast
```

### Broader Closeout Gate

```bash
cargo fmt --check
cargo nextest run -p siumai-spec -p siumai-core -p siumai-protocol-openai --no-fail-fast
```

Use narrower package gates when a slice touches a provider crate and the workspace gate is too large.
Record the reason when a broader gate is deferred.

### Review Gate

Run `review-workstream` before accepting a completed implementation slice, and
`verify-rust-workstream` before marking the lane complete.

## Evidence Anchors

- Stream event contract: `siumai-spec/src/types/streaming.rs`
- Final response assembly: `siumai-core/src/streaming/processor/response_assembly.rs`
- OpenAI completion terminal replay behavior:
  `siumai-provider-openai/src/providers/openai/client/completion.rs`
- OpenAI Responses replay/terminal conversion:
  `siumai-protocol-openai/src/standards/openai/responses_sse`
- Cancellation contract: `siumai-core/src/text.rs`, `siumai-core/src/traits/chat.rs`,
  `siumai/src/text.rs`
- Diagnostics metadata: `siumai-spec/src/types/common.rs`,
  `siumai-spec/src/types/chat/content`
- Error safety: `siumai-core/src/error`, `siumai-core/src/execution/executors/errors.rs`,
  `siumai-core/src/retry_api.rs`
- Tool contracts: `siumai-spec/src/types/tools`, `siumai-spec/src/types/prompt.rs`
- Usage contract: `siumai-spec/src/types/usage.rs`, `siumai-core/src/streaming/processor.rs`
- Unsupported capability contract: `siumai-spec/src/types/common.rs`,
  `siumai-core/src/traits/capabilities.rs`, `siumai-core/src/error/helpers.rs`
- Changelog tracking: `CHANGELOG.md` plus touched crate changelogs.

## Evidence Log

| Date | Task | Command or Evidence | Result | Notes |
| --- | --- | --- | --- | --- |
| 2026-05-26 | AICH-010 | Static audit of the 12 Siumai contract gaps against `repo-ref/ai` and Siumai core/spec/provider files. | Pass | Established scope for this workstream; implementation gates still pending. |
| 2026-05-26 | AICH-010 | `python -m json.tool docs\workstreams\fearless-ai-sdk-contract-hardening\WORKSTREAM.json`; `git diff --check -- CHANGELOG.md docs\workstreams\fearless-ai-sdk-contract-hardening docs\workstreams\INDEX.md`; workstream count check. | Pass | JSON parsed; diff check reported only expected LF-to-CRLF working-copy warnings; index inventory is 90 dirs / 90 status files / 1 active lane. |
| 2026-05-26 | AICH-020 | `cargo fmt --check -p siumai-spec -p siumai-core` | Pass | Full workspace `cargo fmt --check` hit Windows path-length error 206, so formatting was checked on the touched crates. |
| 2026-05-26 | AICH-020 | `cargo nextest run -p siumai-core streaming::processor --no-fail-fast` | Pass | 13 tests passed, including `stream_end_response_content_is_final_replay_not_text_delta`. |
| 2026-05-26 | AICH-020 | `cargo nextest run -p siumai-spec stream --no-fail-fast` | Pass | 15 stream-related spec tests passed. |
| 2026-05-26 | AICH-030 | `cargo fmt --check -p siumai-protocol-openai` | Pass | Formatting passed for the touched protocol crate. |
| 2026-05-26 | AICH-030 | `cargo nextest run -p siumai-protocol-openai responses_stream_preserves_reasoning_delta_and_terminal_only_final_text --no-fail-fast` | No tests | Initial run omitted required `openai-standard,openai-responses` features, so nextest selected 0 tests. Re-run with features passed. |
| 2026-05-26 | AICH-030 | `cargo nextest run -p siumai-protocol-openai --features openai-standard,openai-responses responses_stream_preserves_reasoning_delta_and_terminal_only_final_text --no-fail-fast` | Pass | 1 test passed, proving reasoning deltas plus terminal-only final visible text are preserved. |
| 2026-05-26 | AICH-030 | `cargo nextest run -p siumai-protocol-openai --features openai-standard,openai-responses responses_sse --no-fail-fast` | Pass | 70 Responses SSE tests passed. |
| 2026-05-26 | AICH-040 | `cargo fmt --check -p siumai-core -p siumai` | Pass after import-order fix | Initial run reported only an import ordering diff in `siumai-core/src/text.rs`; after the focused patch, the format gate passed. |
| 2026-05-26 | AICH-040 | `cargo nextest run -p siumai-core stream_with_cancel --no-fail-fast` | Pass | 3 tests passed, including `stream_with_cancel_handle_stops_local_stream_consumption`. |
| 2026-05-26 | AICH-040 | `cargo nextest run -p siumai openai_remote_cancel_propagates_through_siumai_wrapper --no-fail-fast` | Timed out | The broad package invocation exceeded 240s, then 420s during compilation. Re-ran the exact integration-test binary below. |
| 2026-05-26 | AICH-040 | `cargo nextest run -p siumai --test streaming_tests openai_remote_cancel_propagates_through_siumai_wrapper --no-fail-fast` | Pass | 1 OpenAI Responses remote-cancel regression test passed. |
| 2026-05-26 | AICH-050 | `cargo fmt --check -p siumai-spec -p siumai-protocol-openai` | Pass | Touched crates formatted. Full workspace formatting remains avoided because earlier full `cargo fmt` hit Windows path-length error 206. |
| 2026-05-26 | AICH-050 | `cargo nextest run -p siumai-spec metadata --no-fail-fast` | Pass | 22 metadata-related tests passed, including public provider metadata vs private diagnostics classification. |
| 2026-05-26 | AICH-050 | `cargo nextest run -p siumai-spec private_diagnostics --no-fail-fast` | Pass | 8 private diagnostics tests passed for response metadata, raw stream parts, replay raw items, and custom raw/private/diagnostic event-type routing. |
| 2026-05-26 | AICH-050 | `cargo nextest run -p siumai-protocol-openai --features openai-standard compat_stream_unparsable_chunk_emits_raw_error_and_error_finish --no-fail-fast` | Pass | 1 OpenAI-compatible raw chunk fixture passed and now asserts raw chunks are private diagnostics. |
| 2026-05-26 | AICH-060 | `cargo fmt --check -p siumai-core` | Pass | Formatting passed for the touched core error modules. |
| 2026-05-26 | AICH-060 | `cargo nextest run -p siumai-core error --no-fail-fast` | Pass | 26 error-related tests passed, including safe `user_message()` and non-verbose summary diagnostics coverage. |
| 2026-05-26 | AICH-070 | `cargo fmt --check -p siumai-spec -p siumai-core` | Pass | Formatting passed for the touched spec/core crates. Full workspace formatting remains avoided because earlier full `cargo fmt` hit Windows path-length error 206. |
| 2026-05-26 | AICH-070 | `cargo nextest run -p siumai-spec tools --no-fail-fast` | Pass | 42 tool-related tests passed, including validation/fallible-constructor and tool-choice contract coverage. |
| 2026-05-26 | AICH-070 | `cargo nextest run -p siumai-spec prompt_execution_validation --no-fail-fast` | Pass | 3 prompt execution-validation tests passed, including provider-executed true vs false ownership behavior. |
| 2026-05-26 | AICH-070 | `cargo nextest run -p siumai-spec tool_input_start --no-fail-fast` | Pass | 1 spec stream serialization test passed, proving `ToolInputStart` exposes only stable AI SDK fields. |
| 2026-05-26 | AICH-070 | `cargo nextest run -p siumai-core tool_input_start --no-fail-fast` | Pass | 3 core stream tests passed, including stable projection and replay-index exclusion from final tool parts. |
| 2026-05-26 | AICH-070 | `cargo nextest run -p siumai-core streaming::processor --no-fail-fast` | Pass | 14 stream processor tests passed, including the targeted replay-index exclusion regression. |
| 2026-05-26 | AICH-080 | `cargo fmt --check -p siumai-spec -p siumai-core -p siumai-protocol-openai` | Pass | Formatting passed for the touched crates. Full workspace formatting remains avoided because earlier full `cargo fmt` hit Windows path-length error 206. |
| 2026-05-26 | AICH-080 | `cargo nextest run -p siumai-spec usage --no-fail-fast` | Pass | 8 usage-related tests passed, including explicit multi-call aggregation semantics. |
| 2026-05-26 | AICH-080 | `cargo nextest run -p siumai-core streaming::processor --no-fail-fast` | Pass | 15 stream processor tests passed, including repeated cumulative finish usage snapshot replacement. |
| 2026-05-26 | AICH-080 | `cargo nextest run -p siumai-protocol-openai --features openai-standard,openai-responses responses_event_converter_repeated_usage_keeps_latest_snapshot --no-fail-fast` | Pass | 1 OpenAI Responses SSE fixture passed, proving repeated usage events preserve the latest cumulative snapshot through stream processing. |
| 2026-05-26 | AICH-090 | Static comparison with `repo-ref/ai/packages/provider/src/shared/v4/shared-v4-warning.ts` and `repo-ref/ai/packages/provider/src/language-model/v2/language-model-v2-call-warning.ts`. | Pass | AI SDK exposes non-fatal unsupported behavior as warnings; Siumai now adds an explicit reject/warn/provider-fallback policy before warning/error projection. |
| 2026-05-26 | AICH-090 | `cargo fmt --check -p siumai-spec -p siumai-core` | Pass | Formatting passed for the touched spec/core crates. |
| 2026-05-26 | AICH-090 | `cargo nextest run -p siumai-spec unsupported_capability --no-fail-fast` | Pass | 4 unsupported-capability policy serialization and warning-projection tests passed. |
| 2026-05-26 | AICH-090 | `cargo nextest run -p siumai-core unsupported_capability --no-fail-fast` | Pass | 2 runtime policy projection tests passed, covering reject-to-error and warn-to-warning behavior. |
| 2026-05-26 | AICH-090 | `cargo nextest run -p siumai-core reject_if_unsupported --no-fail-fast` | Pass | 2 provider capability helper tests passed, covering supported and missing capability policies. |
| 2026-05-26 | AICH-090 | `cargo nextest run -p siumai-core rerank_guard_blocks_unsupported_provider_before_transform --no-fail-fast` | Pass | 1 executor guard regression stayed green after routing hard capability rejection through the shared policy. |
