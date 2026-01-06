# Public Surface (Facade API) — Stable Paths

This document defines the intended stable public surface of the `siumai` facade crate during the
Alpha.5 split-crate refactor.

Goal: keep the default surface **small, Vercel-aligned, and hard to misuse** (avoid accidental
cross-layer coupling).

## Stability tiers

- **Tier A (stable):** `siumai::prelude::unified::*`
- **Tier B (stable roots, scoped):** `siumai::provider_ext::<provider>::{options,metadata,resources,ext}`
- **Tier C (unstable):** `siumai::experimental::*` (advanced building blocks; may change without notice)

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

For new code, prefer explicit imports from structured submodules:

```rust
use siumai::provider_ext::openai::{metadata::*, options::*};
use siumai::provider_ext::anthropic::{metadata::*, options::*};
```

For navigation/discoverability, each provider extension module may also expose structured submodules:

- `siumai::provider_ext::<provider>::options::*`
- `siumai::provider_ext::<provider>::metadata::*`
- `siumai::provider_ext::<provider>::ext::*`

### 2.1) Protocol mapping (stable facade)

If you need access to protocol-level mapping modules (e.g. for building adapters, fixtures, or
custom providers), use the protocol facade:

```rust
use siumai::protocol::openai::*;
use siumai::protocol::anthropic::*;
use siumai::protocol::gemini::*;
```

These paths should remain stable even if internal protocol crates are renamed during the refactor.

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
