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
- Changelog tracking: `CHANGELOG.md` plus touched crate changelogs.

## Evidence Log

| Date | Task | Command or Evidence | Result | Notes |
| --- | --- | --- | --- | --- |
| 2026-05-26 | AICH-010 | Static audit of the 12 Siumai contract gaps against `repo-ref/ai` and Siumai core/spec/provider files. | Pass | Established scope for this workstream; implementation gates still pending. |
| 2026-05-26 | AICH-010 | `python -m json.tool docs\workstreams\fearless-ai-sdk-contract-hardening\WORKSTREAM.json`; `git diff --check -- CHANGELOG.md docs\workstreams\fearless-ai-sdk-contract-hardening docs\workstreams\INDEX.md`; workstream count check. | Pass | JSON parsed; diff check reported only expected LF-to-CRLF working-copy warnings; index inventory is 90 dirs / 90 status files / 1 active lane. |
| 2026-05-26 | AICH-020 | `cargo fmt --check -p siumai-spec -p siumai-core` | Pass | Full workspace `cargo fmt --check` hit Windows path-length error 206, so formatting was checked on the touched crates. |
| 2026-05-26 | AICH-020 | `cargo nextest run -p siumai-core streaming::processor --no-fail-fast` | Pass | 13 tests passed, including `stream_end_response_content_is_final_replay_not_text_delta`. |
| 2026-05-26 | AICH-020 | `cargo nextest run -p siumai-spec stream --no-fail-fast` | Pass | 15 stream-related spec tests passed. |
