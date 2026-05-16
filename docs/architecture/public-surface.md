# Public Surface (Facade API) — Stable Paths

This document defines the intended stable public surface of the `siumai` facade crate during the
Alpha.5 split-crate refactor.

Goal: keep the default surface **small, Vercel-aligned, and hard to misuse** (avoid accidental
cross-layer coupling).

## Stability tiers

- **Tier A (stable):** `siumai::prelude::unified::*`
- **Tier B (stable roots, scoped):** `siumai::provider_ext::<provider>::{options,metadata,resources,ext}`
- **Tier C (unstable):** `siumai::experimental::*` (advanced building blocks; may change without notice)
- **Compat (time-bounded):** `siumai::compat::*` / `siumai::prelude::compat::*`
  for migration-only builder-style construction.

## Recommended imports

### 1) Unified surface (most code)

Use the Vercel-aligned unified surface as the default:

```rust
use siumai::prelude::unified::*;
```

This is the most stable entrypoint and is designed to cover the 6 stable model families:
Language / Embedding / Image / Rerank / Speech (TTS) / Transcription (STT).

`prelude::unified` intentionally does **not** export compatibility construction aliases such as
`Siumai`, root `Provider`, or deprecated experimental helper aliases. New examples should resolve
models through registry handles or provider config/client APIs, then call family helpers.

During the fearless boundary-convergence workstream, deprecated AI SDK parity aliases and
compatibility helper spellings are kept out of `prelude::unified` unless there is a current runtime
reason to keep them there. Use `siumai::compat::*` or `siumai::prelude::compat::*` for migration-only
imports such as `CallSettings`, `Experimental_*` result aliases,
`experimental_filter_active_tools`, and `step_count_is`. New compatibility aliases should not be
added to the unified prelude without an audit entry and a boundary test.

### 2) Provider-specific APIs (typed options, metadata, resources)

Use provider extension modules (feature-gated):

```rust
use siumai::provider_ext::openai::*;
use siumai::provider_ext::anthropic::*;
use siumai::provider_ext::gemini::*;
```

Vercel-aligned alias (equivalent to `provider_ext`):

```rust
use siumai::providers::openai::*;
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

Provider package helper constructors that return `SiumaiBuilder` bind to the registry-owned builder
type directly; provider extension helpers should not route through the historical
`siumai::provider::*` shim or the removed root `siumai::Provider` alias.

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

These facade modules re-export protocol-owned provider-defined tool constructors. They should not be
implemented in `siumai-core`; core only owns the passive `Tool::ProviderDefined` data shape.

### 4) Non-unified extension capabilities (opt-in)

Some capabilities are intentionally not part of the unified families. Use:

```rust
use siumai::extensions::*;
use siumai::extensions::types::*;
```

This is where non-unified request types live, e.g. `ImageEditRequest` / `ImageVariationRequest`
(used by `ImageExtras`), moderation/file APIs, and provider-specific task types.

File and skill upload helpers are stable explicit modules, not top-level unified prelude names:

```rust
use siumai::files::*;
use siumai::skills::*;
```

The root helpers `siumai::upload_file(...)` and `siumai::upload_skill(...)` remain available for
call-site convenience. `prelude::unified` keeps the `files` and `skills` modules available for
navigation, but it should not directly export `UploadFile*`, `UploadSkill*`, `upload_file`, or
`upload_skill`.

### 5) Registry (provider handle + caching)

If you build multi-provider systems, use the registry surface:

```rust
use siumai::prelude::unified::registry::*;
```

This exports the registry handle types plus `RegistryOptions` for middleware/interceptor setup.
The root `siumai::registry_global` alias has been removed; call `registry::global()` after importing
the scoped registry module, or call `siumai::prelude::unified::registry::global()` explicitly.
Registry contracts such as `ProviderFactory`, `BuildContext`, and `ProviderBuildOverrides` are
scoped under this registry module and should not be imported from the top-level unified prelude.
The root `siumai::provider_catalog::*` mirror has been removed; advanced catalog lookups should
import the owner module `siumai_registry::provider_catalog::*` explicitly.

Registry construction is family-first. Custom factories should implement native family methods such
as `language_model_text_with_ctx(...)`, `embedding_model_family_with_ctx(...)`,
`image_model_family_with_ctx(...)`, and `reranking_model_family_with_ctx(...)`.
`siumai::prelude::unified::registry::*` includes `BuildContext` and
`ProviderBuildOverrides` so custom factory implementations can use the complete family-first
method signatures from the stable registry surface. Generic `LlmClient` factory construction is
compatibility-only and should stay behind explicit `compat_*_client(...)` /
`compat_*_client_with_ctx(...)` methods.

### 6) Low-level / advanced building blocks (opt-in)

For internals (executors, middleware, auth, protocol helpers), use:

```rust
use siumai::experimental::*;
```

Experimental execution hook builders are composition utilities only. Provider-specific request body
presets are not part of the core/facade contract; pass an explicit body builder closure or import a
provider/protocol-owned helper instead.

Low-level streaming converters, factories, encoders, and bridge stream parts are also advanced
integration APIs. Use `siumai::experimental::streaming::*` when building providers, gateways,
transcoders, or stream serializers. `prelude::unified` keeps only stable stream consumption types
such as `ChatStream`, `ChatStreamEvent`, `ChatStreamPart`, and `ChatStreamHandle`.
The root helper `siumai::parse_json_event_stream(...)` remains available for explicit JSON/SSE
parsing, but it is not a top-level `prelude::unified::*` name.

Generic `ClientWrapper` construction is provider-agnostic. Use `ClientWrapper::new(...)` for boxed
advanced clients; provider-named wrapper constructors do not belong in `siumai-core`.

Execution middleware is also an advanced integration API. Import middleware contracts and builders
from `siumai::experimental::execution::middleware::*`, for example
`siumai::experimental::execution::middleware::LanguageModelMiddleware`. `prelude::unified` should
not export middleware internals directly.

Preset middleware helpers must stay provider-agnostic in `siumai-core`. For example,
`ReasoningTagPresets::for_model(...)` returns the generic default tag config; provider-specific
reasoning tag routing belongs in provider/facade extension code that can make an explicit provider
choice. `SystemMessageModeWarningMiddleware` likewise reads only the provider option namespace
given by the caller; automatic middleware wiring passes the configured provider namespace instead
of embedding concrete provider fallbacks in core.

Runtime tool execution helpers follow the AI SDK root-helper shape: stable helper names such as
`tool`, `dynamic_tool`, `ToolExecutionOptions`, `ToolExecutionResult`, `ToolSet`, and
`ExecutableTools` remain available from `prelude::unified::*`. Import the broader runtime module
explicitly when you need less common tool execution contexts or extension points:

```rust
use siumai::tooling::*;
```

`prelude::unified` should not mirror the whole `tooling` module.

Retry policy types and low-level retry helpers are explicit facade runtime controls, not stable
model-family prelude names. Import them from the scoped retry module:

```rust
use siumai::retry_api::*;
```

`prelude::unified` should not directly export `RetryOptions`, `RetryPolicy`, `RetryBackend`,
`BackoffRetryExecutor`, `retry`, `retry_with`, `maybe_retry`, `classify_http_error`,
`backoff_executor_for_provider`, `backoff_options_for_provider`, or `retry_for_provider`.

Error policy helpers are also runtime-owned. `LlmError` remains the shared error data type, while
classification and presentation helpers such as `ErrorCategory`, `is_retryable()`,
`status_code()`, `category()`, `user_message()`, and retry-delay helpers are provided by
`siumai-core::error::LlmErrorExt` and re-exported through `prelude::unified::*`. Direct
`siumai-spec` consumers should treat `LlmError` as data and import runtime policy from `siumai-core`
when they need those helpers.

### 7) Compatibility construction (migration only)

Builder-style construction remains available for migration windows, but it is not part of the
stable unified prelude:

```rust,ignore
use siumai::compat::Siumai;

let client = Siumai::builder()
    .openai()
    .api_key("test-key")
    .model("gpt-4o-mini")
    .build()
    .await?;
```

Provider-specific builder construction is also compatibility-oriented:

```rust,ignore
use siumai::compat::Provider;

let client = Provider::openai()
    .api_key("test-key")
    .model("gpt-4o-mini")
    .build()
    .await?;
```

Use `siumai::prelude::compat::*` only in migration-oriented code that intentionally needs builder
aliases or historical low-level helper aliases alongside stable family types. For example,
`StreamingToolCall*` helpers remain available from `siumai::compat::*` and
`siumai::prelude::compat::*` for source compatibility, but they are not part of
`prelude::unified`. They are no longer re-exported from the facade root; import
`StreamingToolCallDelta`, `StreamingToolCallFunctionDelta`, `StreamingToolCallTracker`,
`StreamingToolCallTrackerOptions`, and `StreamingToolCallTypeValidation` from the explicit compat
surface when migrating older provider-utils style code. Deprecated AI SDK parity names such as
`CallSettings`,
`Experimental_GenerateImageResult`, `Experimental_GeneratedImage`,
`Experimental_LanguageModelStreamPart`, `Experimental_SpeechResult`,
`Experimental_TranscriptionResult`, `ExperimentalLanguageModelStreamPart`,
`experimental_filter_active_tools`, and `step_count_is` also live in the explicit compat surface.
The root `siumai::Provider` path has been removed. Code that intentionally keeps builder-style
construction during migration should import `siumai::compat::Provider` or
`siumai::prelude::compat::Provider` explicitly.
The root `siumai::provider::*` shim has been removed as well. Code that intentionally keeps
builder-style `Siumai` / `SiumaiBuilder` construction during migration should import
`siumai::compat::{Siumai, SiumaiBuilder}` or `siumai::prelude::compat::{Siumai, SiumaiBuilder}`;
new code should prefer `siumai::prelude::unified::registry::*`.
The root `siumai::builder::*` shim has been removed. Code that intentionally needs legacy builder
base internals during migration should import them from `siumai::compat::builder::*`; normal
application code should not depend on builder base types.

The root `siumai::types::*` path has also been removed. Migration code that needs the historical
catch-all type namespace should import `siumai::compat::types::*` or
`siumai::prelude::compat::types::*` explicitly. New code should prefer stable family imports from
`siumai::prelude::unified::*`, extension-only imports from `siumai::extensions::*` /
`siumai::prelude::extensions::*`, and provider-specific data from
`siumai::provider_ext::<provider>::*`.

## Explicitly *not* stable

These top-level module paths are intentionally not part of the stable facade surface:

- `siumai::types::*`
- `siumai::provider::*`
- `siumai::builder::*`
- `siumai::traits::*`
- `siumai::error::*`
- `siumai::streaming::*`
- `siumai::experimental::utils::vertex::*`

They may exist in lower-level crates, but should not be used through the facade.

`siumai::types::*` is a removed historical compatibility path. Use the explicit migration surface
`siumai::compat::types::*` / `siumai::prelude::compat::types::*` only when porting older code that
needs the catch-all namespace. Prefer `siumai::prelude::unified::*` for stable family data,
`siumai::prelude::extensions::*` for non-unified capability types, and provider extension modules
for provider-specific options or metadata. The default `prelude::unified` type exports must stay a
curated explicit list rather than a glob mirror of the broad compatibility type namespace.

Vertex URL helpers are provider-owned. With the `google-vertex` feature enabled, the facade keeps
the compatibility path `siumai::experimental::auth::vertex::*`, which re-exports
`siumai-provider-google-vertex::auth::vertex::*`. Do not import Vertex URL construction helpers from
`siumai-core`.

Google Cloud ADC and service-account auth helpers are also provider-owned. With the `gcp` feature
enabled, the facade keeps `siumai::experimental::auth::{adc,service_account}`, which re-export
`siumai-provider-google-vertex::auth::{adc,service_account}`. `siumai-core::auth` owns only the
generic token-provider contract.

## Related docs

- `docs/workstreams/fearless-boundary-hardening/` (current boundary-hardening checkpoints)
- `docs/workstreams/fearless-spec-core-boundary-convergence/` (current spec/core/facade convergence workstream)
- `docs/architecture/provider-extensions.md` (how provider-specific features work)
- `docs/migration/migration-0.11.0-beta.6.md` (family APIs + compat surface)
- `docs/migration/migration-0.11.0-beta.5.md` (split-crate breaking changes and migration cookbook)
