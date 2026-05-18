# GIR-040 Response Parsing

Date: 2026-05-18

## Summary

Completed the Google Interactions response parsing slice in `siumai-provider-gemini`.

## What Changed

- Added `providers/gemini/interactions/response.rs`.
- Parsed completed Interactions responses into stable `ChatResponse` values.
- Covered:
  - completed text output,
  - usage mapping,
  - finish reason mapping,
  - service tier and interaction metadata,
  - reasoning signatures,
  - function calls,
  - built-in tool calls/results,
  - image blocks,
  - source extraction.

## Validation

- `cargo nextest run -p siumai-provider-gemini --all-features google_interactions_response --no-fail-fast`
- `cargo fmt -p siumai-provider-gemini -- --check`
- `cargo nextest run -p siumai-provider-gemini --all-features --no-fail-fast`
- `cargo clippy -p siumai-provider-gemini --all-features --all-targets -- -D warnings`
- `git diff --check`

## Next

GIR-050: implement non-stream model-mode execution and background polling.
