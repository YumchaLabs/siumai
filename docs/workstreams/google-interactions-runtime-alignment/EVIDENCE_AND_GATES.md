# Google Interactions Runtime Alignment - Evidence And Gates

Status: Active
Last updated: 2026-05-18

## Smallest Current Repro

The currently shipped Interactions handle is package-visible but fail-fast:

```bash
cargo nextest run -p siumai-provider-gemini --all-features interactions_handle_is_explicitly_deferred_at_runtime --no-fail-fast
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
