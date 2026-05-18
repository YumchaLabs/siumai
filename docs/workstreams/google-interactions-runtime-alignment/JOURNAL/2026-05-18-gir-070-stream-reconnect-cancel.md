# GIR-070 Stream Reconnect And Cancel

Date: 2026-05-18
Status: Done

## Summary

Implemented Google Interactions agent streaming runtime parity with the AI SDK reference:

- agent streaming now posts `/interactions` with `background: true`;
- non-terminal background interactions stream through `GET /interactions/{id}?stream=true`;
- unexpected stream closure reconnects with `last_event_id` sourced from the JSON `event_id`;
- empty stream attempts obey a bounded retry budget;
- cancellable agent streams send best-effort `POST /interactions/{id}/cancel`;
- model-mode streaming continues to use `POST /interactions` with `stream: true`.

Added a core transport seam, `HttpTransport::execute_get_stream`, so provider-owned GET SSE
endpoints can be tested with no network access.

## Validation

- `cargo nextest run -p siumai-provider-gemini --all-features google_interactions_stream_reconnect --no-fail-fast` passed.
- `cargo nextest run -p siumai-provider-gemini --all-features google_interactions_stream --no-fail-fast` passed.
- `cargo fmt -p siumai-provider-gemini -- --check` passed.
- `cargo fmt -p siumai-core -- --check` passed.
- `cargo clippy -p siumai-provider-gemini --all-features --all-targets -- -D warnings` passed.
- `cargo nextest run -p siumai-provider-gemini --all-features --no-fail-fast` passed.
- `git diff --check` passed.

## Follow-Up

GIR-080 should update public facade parity tests now that agent streaming is no longer an explicit
deferred boundary.
