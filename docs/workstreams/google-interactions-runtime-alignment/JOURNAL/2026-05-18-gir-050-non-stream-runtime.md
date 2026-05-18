# GIR-050 Non-Stream Runtime

Date: 2026-05-18
Status: DONE

## Summary

Implemented provider-owned non-stream execution for `google.interactions(...)`.

The runtime now:

- posts model and agent requests to `/interactions`;
- sends `Api-Revision: 2026-05-20`;
- parses terminal Interactions responses through `interactions/response.rs`;
- polls background agent responses with `GET /interactions/{id}`;
- honors `providerOptions.google.pollingTimeoutMs`;
- preserves warnings plus request/response envelopes on `ChatResponse`;
- keeps streaming explicitly deferred for GIR-060/GIR-070.

## Files

- `siumai-provider-gemini/src/providers/gemini/interactions.rs`
- `siumai-provider-gemini/src/providers/gemini/interactions/runtime.rs`
- `siumai-provider-gemini/src/providers/gemini/builder.rs`
- `docs/workstreams/google-interactions-runtime-alignment/*`

## Validation

- `cargo nextest run -p siumai-provider-gemini --all-features google_interactions_non_stream --no-fail-fast`
- `cargo fmt -p siumai-provider-gemini -- --check`
- `cargo clippy -p siumai-provider-gemini --all-features --all-targets -- -D warnings`
- `cargo nextest run -p siumai-provider-gemini --all-features --no-fail-fast`

## Next

GIR-060 should implement Interactions SSE event conversion. Do not route Interactions streams through
ordinary Gemini `:streamGenerateContent`; the wire shape is owned by the Interactions API.
