# GIR-060 Stream Runtime

Date: 2026-05-18
Status: DONE

## Summary

Implemented provider-owned Google Interactions model-mode streaming under
`siumai-provider-gemini/src/providers/gemini/interactions/stream.rs`.

The runtime now posts model-mode `google.interactions(...)` stream requests to
`/v1beta/interactions` with `stream: true`, sends `Api-Revision: 2026-05-20`, uses the shared HTTP
execution layer/custom transport path, and converts Interactions SSE events into stable
`ChatStreamPart` values.

Covered stream events:

- `interaction.created` -> response metadata;
- `step.start`/`step.delta`/`step.stop` for text;
- reasoning summaries and final signatures;
- function-call argument deltas and final tool calls;
- built-in tool results plus deduped source parts;
- image deltas as file parts;
- `interaction.completed` -> usage, finish reason, interaction id, service tier, and final
  `StreamEnd` metadata.

Agent-mode streaming still fails fast. Reconnect with `last_event_id` and cancel-on-abort are
explicitly left for GIR-070.

## Validation

- `cargo nextest run -p siumai-provider-gemini --all-features google_interactions_stream --no-fail-fast`
- `cargo fmt -p siumai-provider-gemini -- --check`
- `cargo clippy -p siumai-provider-gemini --all-features --all-targets -- -D warnings`
- `cargo nextest run -p siumai-provider-gemini --all-features --no-fail-fast`

All passed on 2026-05-18.

