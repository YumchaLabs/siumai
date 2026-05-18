# GIR-080 Public Path Parity

Date: 2026-05-18

## Summary

Replaced the obsolete public-path fail-fast guard with facade-level no-network parity tests for the
implemented Google Interactions runtime.

## Changes

- Added a test-local Interactions capture transport to `siumai/tests/provider_public_path_parity_test.rs`
  with JSON POST, streaming POST, and streaming GET support.
- Proved model-mode non-stream calls from `Provider::google()`, `provider_ext::google`, and direct
  `GoogleInteractionsLanguageModel` handles post to `/v1beta/interactions` and parse stable
  `ChatResponse` metadata.
- Proved model-mode streaming public paths post to `/v1beta/interactions` with `stream: true`.
- Proved agent-mode streaming public paths create a background interaction and then stream through
  `GET /interactions/{id}?stream=true`, including `last_event_id` reconnect behavior.

## Validation

- `cargo nextest run -p siumai --features google --test provider_public_path_parity_test google_interactions --no-fail-fast`
- `cargo nextest run -p siumai --features google --test provider_public_path_parity_test --no-fail-fast`
- `cargo fmt -p siumai -- --check`
- `git diff --check`
- `cargo nextest run -p siumai-provider-gemini --all-features --no-fail-fast`

## Notes

The public gate uses `--features google`; `google_interactions` is a test-name filter, not a Cargo
feature in this workspace.
