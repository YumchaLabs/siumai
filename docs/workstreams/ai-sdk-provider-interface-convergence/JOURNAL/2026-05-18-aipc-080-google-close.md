# AIPC-080 Google Close

Date: 2026-05-18

## Summary

Closed the Google/Gemini package-surface row by separating import/construction parity from runtime
execution. The `google.interactions(...)` surface is intentionally visible and fail-fast in Siumai
today, while real `/v1beta/interactions` request conversion, polling, cancellation, signature
round-trips, interaction-id compaction, and stream transforms move to
`docs/workstreams/google-interactions-runtime-alignment`.

## Evidence

- `cargo nextest run -p siumai-provider-gemini --all-features interactions_handle_is_explicitly_deferred_at_runtime --no-fail-fast`
- `cargo nextest run -p siumai --features google google_interactions_package_surface_is_explicitly_deferred_from_chat_runtime --test provider_public_path_parity_test --no-fail-fast`

## Decision

Treat Interactions runtime as a child execution lane, not a package parity cleanup item. This keeps
ordinary Gemini `:generateContent` behavior separate from the dedicated Interactions wire contract
and lets AIPC-080 close without hiding a runtime gap.
