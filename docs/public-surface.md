# Public Surface (Facade API) — Stable Paths

This document defines the intended stable public surface of the `siumai` facade crate during the
Alpha.5 split-crate refactor.

Goal: keep the default surface **small, Vercel-aligned, and hard to misuse** (avoid accidental
cross-layer coupling).

## Recommended imports

### 1) Unified surface (most code)

Use the Vercel-aligned unified surface as the default:

```rust
use siumai::prelude::unified::*;
```

This is the most stable entrypoint and is designed to cover the 6 stable model families:
Language / Embedding / Image / Rerank / Speech (TTS) / Transcription (STT).

### 2) Provider-specific APIs (typed options, metadata, resources)

Use provider extension modules (feature-gated):

```rust
use siumai::provider_ext::openai::*;
use siumai::provider_ext::anthropic::*;
use siumai::provider_ext::gemini::*;
```

### 3) Provider-hosted tools (provider-executed tools)

Use hosted tools via the stable module path:

```rust
use siumai::hosted_tools::openai as openai_tools;
use siumai::hosted_tools::anthropic as anthropic_tools;
use siumai::hosted_tools::google as google_tools;
```

### 4) Non-unified extension capabilities (opt-in)

Some capabilities are intentionally not part of the unified families. Use:

```rust
use siumai::extensions::*;
use siumai::extensions::types::*;
```

This is where non-unified request types live, e.g. `ImageEditRequest` / `ImageVariationRequest`
(used by `ImageExtras`), moderation/file APIs, and provider-specific task types.

### 5) Registry (provider handle + caching)

If you build multi-provider systems, use the registry surface:

```rust
use siumai::prelude::unified::registry::*;
```

This exports the registry handle types plus `RegistryOptions` for middleware/interceptor setup.

### 6) Low-level / advanced building blocks (opt-in)

For internals (executors, middleware, auth, protocol helpers), use:

```rust
use siumai::experimental::*;
```

## Explicitly *not* stable

These top-level module paths are intentionally not part of the stable facade surface:

- `siumai::types::*`
- `siumai::traits::*`
- `siumai::error::*`
- `siumai::streaming::*`

They may exist in lower-level crates (e.g. `siumai-core`) but should not be used through the facade.

## Related docs

- `docs/next-steps.md` (refactor checkpoints and guardrails)
- `docs/provider-extensions.md` (how provider-specific features work)
- `docs/migration-0.11.0-beta.5.md` (breaking changes and migration cookbook)
