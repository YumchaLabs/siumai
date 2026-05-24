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

This is the most stable entrypoint and is designed to cover the 7 stable model families:
Language / Embedding / Image / Rerank / Speech (TTS) / Transcription (STT) / Video.

`prelude::unified` intentionally does **not** export compatibility construction aliases such as
`Siumai`, root `Provider`, or deprecated experimental helper aliases. New examples should resolve
models through registry handles or provider config/client APIs, then call family helpers.

During the fearless boundary-convergence workstream, deprecated AI SDK parity aliases and
compatibility helper spellings are kept out of `prelude::unified` unless there is a current runtime
reason to keep them there. Use `siumai::compat::*` or `siumai::prelude::compat::*` for migration-only
imports such as `CallSettings`, `Experimental_*` result aliases,
`experimental_filter_active_tools`, and `step_count_is`. New compatibility aliases should not be
added to the unified prelude without an audit entry and a boundary test.

Legacy `ContentPart` is not part of this stable unified prelude. It remains available only through
`siumai::compat::content::*` / `siumai::prelude::compat::content::*` for serde and migration code.
Use request-directional prompt parts (`UserContentPart`, `AssistantContentPart`, `ToolContentPart`)
or response-directional generated output parts (`GenerateTextContentPart`, `TextOutput`,
`ReasoningOutput`, `FileOutput`, `Source`) for new examples.

The facade also exposes directional content namespaces for code that wants imports to describe the
data flow explicitly:

```rust
use siumai::content::prompt::*;
use siumai::content::output::*;
use siumai::content::compat::*;
```

`prelude::unified` keeps the `prompt` and `output` navigation modules available as
`siumai::prelude::unified::{prompt, output}`. It intentionally does not expose
`content::compat` there; migration code should import legacy payloads from `siumai::compat::content`
or `siumai::content::compat`.

#### Response parsing, compatibility payloads, and generated output

Provider and protocol response parsers may still populate `ChatResponse` / `MessageContent` because
those serialized payloads are part of the legacy compatibility contract. That does **not** make
`ContentPart` the canonical response model. Parser code that must construct legacy response content
should do so through a local `response_content` module or equivalent compatibility adapter. These
adapters own empty request-side `provider_options` defaults and preserve response-side
`provider_metadata`.

Generated-output projection is a separate, fallible boundary. Use `GenerateTextContentPart` and the
output-part carriers for response shapes that can be represented without losing provider data. Do
not force hosted tool results, approval requests, files, images, audio, or provider-specific
metadata through generated-output projection unless the losslessness has been proven. Those shapes
should remain `ChatResponse` compatibility payloads until an ADR-backed public output model exists.

Bridge response/stream APIs are primitive serialization boundaries: they accept `ChatResponse` /
`ChatStreamEvent` and encode target JSON/SSE. They should not import parser-local
`response_content` adapters or spec-owned generated-output projection helpers.

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

Typed request options are provider-owned. Typed response metadata may be protocol-owned when it is a
projection of provider wire-format semantics; provider extension modules keep stable re-export paths
so application imports do not need to distinguish the internal owner crate.

For navigation/discoverability, each provider extension module may also expose structured submodules:

- `siumai::provider_ext::<provider>::options::*`
- `siumai::provider_ext::<provider>::metadata::*`
- `siumai::provider_ext::<provider>::legacy_params::*` (migration-only client-level defaults)
- `siumai::provider_ext::<provider>::ext::*`

Legacy provider parameter structs such as `OpenAiParams`, `AnthropicParams`, and `GeminiParams`
are client-level default carriers from older APIs. Keep them out of the flattened provider
extension root and import them through `legacy_params::*` only when migrating code that still calls
provider constructors requiring those parameter structs.

`siumai::provider_ext::google` is the Google package facade over the Gemini runtime. It intentionally
mirrors the audited `provider_ext::gemini` surface while owning Google-named builder helpers such as
`google()` and `create_google()`. Migration code that imports Gemini-era parameter structs through
the Google package path should use `siumai::provider_ext::google::legacy_params::*`; do not flatten
those legacy parameters into the Google extension root.

Provider package helper constructors that return `SiumaiBuilder` bind to the registry-owned builder
type directly; provider extension helpers should not route through the historical
`siumai::provider::*` shim or the removed root `siumai::Provider` alias.

OpenAI-compatible provider-list generation is provider-owned infrastructure. Import
`siumai_provider_openai_compatible::siumai_for_each_openai_compatible_provider` directly when
generating registry/provider glue; the facade root does not re-export this macro.

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
(used by `ImageExtras`), moderation/file APIs, and provider-specific task types. Video's stable
family surface is `siumai::video::*` / `VideoModel`; the low-level `VideoGenerationCapability`
and its task payloads remain here only for provider adapters and compatibility code.
Music generation remains extension-only: use `siumai::extensions::MusicGenerationCapability` or
provider extension modules, not a stable `MusicModel` family.

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
The historical `siumai::prelude::registry::*` mirror has been removed; import
`siumai::prelude::unified::registry::*` or `siumai::registry::*` explicitly.
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
`compat_*_client_with_ctx(...)` methods. Downstream code that still needs the generic client types
should import them from `siumai::compat::client::{LlmClient, ClientWrapper}`; the old
`siumai::experimental::client` path remains an advanced alias during migration.
Registry execution now uses narrower facets derived from that custom-provider contract:
`ProviderFamilyFactory` for stable family handles, `ProviderCompatibilityFactory` for legacy
generic-client entry points, and `ProviderExtensionFactory` for non-family extension capabilities.
These facets prevent the stable registry handles from depending on the wide compatibility surface.
Custom registry/factory code that must name the generic client trait should use
`siumai_registry::compat::client::LlmClient`; the old `siumai_registry::LlmClient` root import is
no longer part of the small registry root surface.
The old `siumai_registry::registry::factory::build_*_client(...)` helpers are compatibility-only
shims; new registry/provider work should implement `ProviderFactory::*_family_with_ctx(...)` using
provider-owned config builders instead of calling those broad generic-client constructors.
OpenAI-compatible vendor or dynamic provider ids should use
`openai_compatible_provider_factory(...)` instead of concrete OpenAI-compatible factory
construction.
Built-in provider factories should keep typed-client construction and typed-client `Arc` projection
as local helpers. Stable family and explicit compatibility methods can share those helpers, but the
public surface should not grow new broad generic-client aliases to compensate for duplicated factory
glue.
Azure's deployment-based URL mode is the current provider-specific exception; use the registry
helper `azure_provider_factory_with_options(...)` instead of concrete Azure factory construction.

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

Low-level utility helpers are explicit root imports, not default unified-prelude names.
Provider-utils helpers that remain at the root are backed by `siumai-provider-utils`, not the old
`siumai-core::utils` owner path. Helpers such as download helpers, header normalization,
environment setting loaders, JSON parsing/instruction helpers, provider-option/reference parsers,
URL support maps, base64/data helpers, reasoning mapping helpers, media helpers, and runtime type
validators remain available for opt-in utility users:

```rust
use siumai::{parse_json, normalize_headers};
```

`prelude::unified` does not export broad provider-utils helper groups. It keeps the
application-facing AI SDK helper layer, including schema helpers,
ID generation helpers, stop-condition helpers, UI part predicates, `SerialJobExecutor`, and
`ToolNameMapping`, without mirroring the whole `siumai-provider-utils` helper set or the historical
`siumai-core::utils` compatibility module. The stable prelude intentionally does not export broad
utility groups such as `Arrayable`/nullability helpers, base64/data helpers, prompt `DataContent`
projection helpers, download/header/settings helpers, JSON parse/instruction helpers, reasoning
mapping helpers, runtime user-agent/version helpers, URL support helpers, or media helpers; import
those from the root facade explicitly.

Retained broad exports are limited to explicit namespaces where the namespace itself states the
boundary and avoids accidental root/prelude coupling:

- `siumai::protocol::<provider>::*` is the stable protocol-mapping facade for adapters, fixtures,
  and custom providers.
- `siumai::hosted_tools::<provider>::*` is the stable provider-hosted tool namespace.
- `siumai::content::{prompt,output,compat}::*` is directional content navigation; `compat` remains
  explicit because legacy chat payloads are migration-only.
- `siumai::prelude::compat::{types,content}::*` is the time-bounded compatibility namespace for
  historical broad imports.
- `siumai::experimental::{streaming,execution,providers,standards,...}::*` remains advanced and
  unstable; use it only when implementing providers, gateways, or migration tooling.

Generic `ClientWrapper` construction is provider-agnostic. Use
`siumai::compat::client::ClientWrapper::new(...)` for boxed advanced clients; provider-named wrapper
constructors do not belong in `siumai-core`.

UI message validation and conversion helpers live under the explicit facade module:

```rust
use siumai::ui::*;
```

`siumai::ui` exports the stable conversion/validation surface only. It should not mirror the entire
`siumai-core::ui` module by wildcard; new core-local UI helpers need an intentional facade export
decision before becoming public through `siumai`.

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

Legacy chat content carriers are also compatibility-only. Migration code that intentionally needs
the serde-compatible `ContentPart` / `MessageContent` payload should import
`siumai::compat::content::{ContentPart, MessageContent}` or
`siumai::prelude::compat::content::*`. `prelude::unified` no longer exports legacy `ContentPart`;
this is an intentional namespace break so new code does not see the dual request/response carrier
as the canonical content model. New request examples should use `ModelMessage`, `UserContentPart`,
`AssistantContentPart`, and `ToolContentPart`; new response examples should use
`GenerateTextContentPart` and output-part carriers. Prefer `siumai::content::prompt::*` and
`siumai::content::output::*` when you want directionally named imports; use
`siumai::content::compat::*` only for compatibility payloads.

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
