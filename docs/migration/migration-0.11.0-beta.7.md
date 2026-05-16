# Migration Guide: `0.11.0-beta.6` -> `0.11.0-beta.7`

This guide covers the user-visible migration points for `0.11.0-beta.7`.
Most applications that only call chat/text helpers can upgrade by changing the Cargo version.
The main source changes affect advanced users who implement model traits, inspect streaming events,
construct shared structs directly, or compare serialized snapshots.

## TL;DR

- Normal text/chat callers: usually no source migration.
- Model trait generics: use the canonical family traits (`TextModel`, `EmbeddingModel`,
  `ImageModel`, `RerankingModel`, `SpeechModel`, `TranscriptionModel`, `VideoModel`).
- Generic `LlmClient` paths: keep them only for migration or extension-only integrations; use
  registry family handles and `*_family_with_ctx(...)` factory methods for new code.
- Registry global handle: call `registry::global()` or
  `siumai::prelude::unified::registry::global()`; the root `siumai::registry_global` alias was
  removed.
- Provider catalog: import advanced provider catalog lookups from
  `siumai_registry::provider_catalog::*`; the root `siumai::provider_catalog::*` mirror was
  removed.
- OpenAI-compatible provider-list macro: import
  `siumai_provider_openai_compatible::siumai_for_each_openai_compatible_provider` directly; the
  facade root re-export was removed.
- `ClientWrapper`: use `ClientWrapper::new(...)`; provider-named wrapper constructors were removed
  from `siumai-core`.
- Streaming consumers: prefer semantic accessors such as `event.text_delta()` and
  `event.part_ref()`.
- Advanced middleware users: import `LanguageModelMiddleware` from
  `siumai::experimental::execution::middleware`, not `prelude::unified`.
- Reasoning middleware presets: `ReasoningTagPresets::for_model(...)` now returns the
  provider-agnostic default tag config; choose provider-specific tags with `with_tag(...)`.
- System-message warning middleware: construct
  `SystemMessageModeWarningMiddleware::new("provider-id")` with the provider option namespace to
  inspect; core no longer hard-codes provider fallback namespaces.
- StreamingToolCall* helpers: import `StreamingToolCallTracker` and related low-level delta helpers
  from `siumai::compat`, not from the facade root or `prelude::unified`.
- Provider builder entry: import `Provider` from `siumai::compat` or `siumai::prelude::compat`;
  the root `siumai::Provider` alias was removed. Import `Siumai` / `SiumaiBuilder` from
  `siumai::compat` or `siumai::prelude::compat`; the root `siumai::provider::*` shim was removed.
  Legacy builder base internals moved to `siumai::compat::builder::*`; the root
  `siumai::builder::*` shim was removed.
- Root broad type namespace: import migration-only catch-all types from
  `siumai::compat::types::*` or `siumai::prelude::compat::types::*`; the root `siumai::types::*`
  path was removed.
- Deprecated AI SDK parity aliases: import `CallSettings`, `Experimental_*` result aliases,
  `experimental_filter_active_tools`, and `step_count_is` from `siumai::compat` when needed.
- File/skill upload helpers: import upload helper types from `siumai::files::*` /
  `siumai::skills::*` instead of relying on top-level `prelude::unified::*`.
- JSON/SSE stream parsing: call `siumai::parse_json_event_stream(...)` explicitly instead of
  importing it from `prelude::unified::*`.
- Low-level utility helpers: import download/header/setting/JSON parse/provider-option helper names
  from the facade root, e.g. `siumai::{parse_json, normalize_headers}`, not
  `prelude::unified::*`.
- Retry API controls: import `RetryOptions`, `RetryPolicy`, `retry_with`, and related helpers from
  `siumai::retry_api`, not from `prelude::unified::*`.
- Error policy helpers: if you call `is_retryable()`, `status_code()`, `category()`,
  `user_message()`, or retry-delay helpers on `LlmError`, import the core-owned extension trait
  `siumai::prelude::unified::LlmErrorExt` or `siumai_core::error::LlmErrorExt`.
- Anthropic prompt-cache and document-part message helpers: prefer the provider-owned
  `AnthropicChatMessageExt` trait from `siumai::provider_ext::anthropic::options`.
  Historical `ChatMessageBuilder` Anthropic helper methods were removed from `siumai-spec`.
- `HttpConfig` defaults: `siumai-spec` no longer reads process environment variables. Runtime
  builder/config-first paths still resolve the `SIUMAI_STREAM_DISABLE_COMPRESSION` default in
  `siumai-core`; direct `HttpConfig::default()` construction is now a deterministic data default.
- Usage and metadata snapshots: expect AI SDK-aligned field names and provider-rooted metadata.
- Raw provider envelopes: opt in through request/response retention controls instead of parsing
  debug output.
- Custom `ProviderSpec` implementations: replace string-returning `*_url(...)` hooks with
  fallible `try_*_url(...)` hooks.

If you are upgrading from `0.11.0-beta.5` or earlier, read this guide after
`docs/migration/migration-0.11.0-beta.6.md`.

## 1) Model-family traits

`0.11.0-beta.7` finishes the trait-name cleanup started in the previous beta.
The stable Rust family names are now:

- `TextModel`
- `EmbeddingModel`
- `ImageModel`
- `RerankingModel`
- `SpeechModel`
- `TranscriptionModel`
- `VideoModel`

Use these names for application-level generics and helper functions.

```rust,ignore
use siumai::prelude::unified::{ChatRequest, TextModel, text};

async fn run_text<M: TextModel + ?Sized>(model: &M, request: ChatRequest) -> anyhow::Result<()> {
    let response = text::generate(model, request, text::GenerateOptions::default()).await?;
    println!("{}", response.text);
    Ok(())
}
```

If you specifically need a full language model handle with metadata, `LanguageModel` remains
available:

```rust,ignore
use siumai::prelude::unified::LanguageModel;

fn accepts_language_model<M: LanguageModel + ?Sized>(_model: &M) {}
```

`LanguageModelV4`, `ImageModelV4`, and `VideoModelV4` are provider-contract markers for
AI SDK-aligned low-level integrations. Most application code should not use them unless an
explicit helper requires them.

## 2) Generic `LlmClient` compatibility paths

Registry construction is now family-first. New factories should implement the family methods such as
`language_model_text_with_ctx(...)`, `embedding_model_family_with_ctx(...)`,
`image_model_family_with_ctx(...)`, `reranking_model_family_with_ctx(...)`,
`speech_model_family_with_ctx(...)`, `transcription_model_family_with_ctx(...)`, and
`video_model_family_with_ctx(...)`.

`BuildContext` and `ProviderBuildOverrides` are available from
`siumai::prelude::unified::registry::*`, so custom factory implementations do not need to import
private registry internals to implement those `*_with_ctx(...)` methods. `ProviderFactory` is also
scoped to `siumai::prelude::unified::registry::*`; it is not exported from the top-level unified
prelude.

Generic `LlmClient` factory construction remains available only through explicit
`compat_*_client(...)` / `compat_*_client_with_ctx(...)` methods. Use those methods for migration
code or for extension-only surfaces that do not yet have a first-class model family.

`ClientWrapper` is now provider-agnostic in `siumai-core`. Provider-named convenience constructors
such as `ClientWrapper::openai(...)` were removed; use `ClientWrapper::new(...)` instead.

Before:

```rust,ignore
let wrapper = ClientWrapper::openai(Box::new(client));
```

After:

```rust,ignore
let wrapper = ClientWrapper::new(Box::new(client));
```

Before:

```rust,ignore
let client = factory.compat_language_client("gpt-4o-mini").await?;
let chat = client.as_chat_capability().expect("chat support");
```

After:

```rust,ignore
let model = factory
    .language_model_text_with_ctx("gpt-4o-mini", &build_context)
    .await?;
```

Application code should usually resolve models through the registry facade instead of calling a
factory directly:

```rust,ignore
use siumai::prelude::unified::*;

let model = registry::global().language_model("openai:gpt-4o-mini")?;
```

The extra root alias `siumai::registry_global` has been removed. If older code used it, call the
scoped registry module instead:

```rust,ignore
let model = siumai::prelude::unified::registry::global()
    .language_model("openai:gpt-4o-mini")?;
```

The facade no longer mirrors `siumai-registry`'s provider catalog at
`siumai::provider_catalog::*`. If advanced catalog code used that path, import the registry-owned
module explicitly:

```rust,ignore
use siumai_registry::provider_catalog::{get_provider_info, get_supported_providers};
```

The facade no longer re-exports the OpenAI-compatible provider-list macro. This macro is generation
infrastructure for registry/provider glue, not an application facade API.

Before:

```rust,ignore
use siumai::siumai_for_each_openai_compatible_provider;
```

After:

```rust,ignore
use siumai_provider_openai_compatible::siumai_for_each_openai_compatible_provider;
```

## 2.1) Anthropic Message Helpers

Anthropic prompt caching and document-part options are now available through a provider-owned
extension trait. This keeps provider-specific request helpers out of the stable spec chat builder
for new code.

Before:

```rust,ignore
let system = ChatMessage::system("large reusable context")
    .cache_control(CacheControl::Ephemeral)
    .build();

let user = ChatMessage::user("summarize this")
    .with_file_url("https://example.com/doc.pdf", "application/pdf")
    .anthropic_document_citations_for_part(1, true)
    .build();
```

After:

```rust,ignore
use siumai::provider_ext::anthropic::options::AnthropicChatMessageExt;

let system = ChatMessage::system("large reusable context")
    .build()
    .with_anthropic_cache_control(CacheControl::Ephemeral);

let user = ChatMessage::user("summarize this")
    .with_file_url("https://example.com/doc.pdf", "application/pdf")
    .build()
    .with_anthropic_document_citations_for_part(1, true);
```

## 3) Streaming events

The canonical stream payload is now `ChatStreamPart`, carried by
`ChatStreamEvent::Part` / `ChatStreamEvent::PartWithReplay`. Application code should prefer
semantic accessors so it stays compatible as more AI SDK stream parts are exposed.

Before:

```rust,ignore
match event {
    ChatStreamEvent::TextDelta { delta, .. } => print!("{delta}"),
    _ => {}
}
```

After:

```rust,ignore
if let Some(delta) = event.text_delta() {
    print!("{delta}");
}
```

For richer stream consumers:

```rust,ignore
if let Some(delta) = event.reasoning_delta() {
    eprintln!("reasoning: {delta}");
}

if let Some(part) = event.part_ref() {
    // Inspect source, tool, metadata, finish, custom, file, and reasoning-file parts here.
}
```

## 4) Usage values

`Usage` now keeps AI SDK-style `inputTokens` / `outputTokens` as the canonical stable layer while
legacy prompt/completion/total counts remain available through accessors and compatibility serde.

Avoid struct literals or direct legacy field reads.

Before:

```rust,ignore
let prompt = usage.prompt_tokens;
let completion = usage.completion_tokens;
```

After:

```rust,ignore
let prompt = usage.prompt_tokens();
let completion = usage.completion_tokens();
let total = usage.total_tokens();

let input = &usage.input_tokens;
let output = &usage.output_tokens;
```

Construction should use constructors or the builder:

```rust,ignore
let usage = Usage::new(12, 7);

let usage = Usage::builder()
    .with_input_total_tokens(12)
    .with_output_text_tokens(7)
    .build();
```

## 5) Serialized snapshots and metadata

If your tests compare JSON snapshots, update them for the AI SDK-aligned shapes.
Common differences are:

- `ToolChoice` and `FinishReason` serialize with AI SDK public values.
- `Warning` now has explicit compatibility/deprecated/unsupported shapes.
- `Usage` prefers `inputTokens`, `outputTokens`, and `raw`.
- `ResponseMetadata` serializes `modelId` / `timestamp`.
- `providerMetadata` is provider-rooted (`provider id -> object`) across response and stream
  metadata.
- Stream parts may appear as stable `ChatStreamPart` values rather than provider-specific custom
  payloads.

## 6) Raw provider request/response envelopes

For audit, debugging, or gateway work, use the new request/response body retention controls exposed
by the AI SDK-style helper options. Do not parse formatted debug strings; inspect the structured
metadata and retained body values instead.

## 7) File and skill upload helper imports

File and skill upload helpers are explicit non-family modules. If code imported helper types or
functions from the top-level unified prelude, import them from their owning modules instead.

Before:

```rust,ignore
use siumai::prelude::unified::{UploadFileOptions, UploadSkillOptions, upload_file, upload_skill};
```

After:

```rust,ignore
use siumai::files::{self, UploadFileOptions};
use siumai::skills::{self, UploadSkillOptions};

// Or keep the root helper functions for call-site convenience:
let _ = siumai::upload_file;
let _ = siumai::upload_skill;
```

`prelude::unified` still exposes the `files` and `skills` modules for navigation, but direct
`UploadFile*`, `UploadSkill*`, `upload_file`, and `upload_skill` names are not part of the stable
family prelude.

## 8) JSON/SSE parser imports

`parse_json_event_stream(...)` remains available as an explicit root helper for byte streams that
carry SSE `data:` JSON payloads. It is no longer a top-level unified prelude import.

Before:

```rust,ignore
use siumai::prelude::unified::parse_json_event_stream;
```

After:

```rust,ignore
let stream = siumai::parse_json_event_stream(byte_stream);
```

Advanced stream integration code can also import lower-level streaming utilities from
`siumai::experimental::streaming::*`.

## 9) Low-level utility helper imports

The unified prelude no longer mirrors low-level utility helpers from `siumai-core::utils`. These
helpers remain available as explicit facade root imports for advanced utility users.

Before:

```rust,ignore
use siumai::prelude::unified::{parse_json, normalize_headers, load_api_key};
```

After:

```rust,ignore
use siumai::{parse_json, normalize_headers, load_api_key};
```

This applies to download helpers, header normalization, environment setting loaders, JSON
instruction/parse helpers, provider-option/reference parsers, URL support helpers, and runtime type
validators. Application-facing helper names such as `json_schema`, `generate_id`,
`create_id_generator`, `has_tool_call`, `filter_active_tools`, UI part predicates,
`SerialJobExecutor`, and `ToolNameMapping` remain in `prelude::unified`.

## 10) Retry API imports

Retry options and retry helper functions remain stable, but they are now scoped to the explicit
runtime module instead of the stable family prelude.

Before:

```rust,ignore
use siumai::prelude::unified::{RetryOptions, RetryPolicy, retry_with};
```

After:

```rust,ignore
use siumai::retry_api::{RetryOptions, RetryPolicy, retry_with};
```

Per-call family options still accept `RetryOptions`; only the import path for direct retry API
controls changes.

## 11) Custom `ProviderSpec` route hooks

The historical string-returning `ProviderSpec` route hooks were removed:
`chat_url`, `embedding_url`, `image_url`, `image_edit_url`, `image_variation_url`, `rerank_url`,
`models_url`, and `model_url`.

Implement the matching fallible hook instead. This keeps unsupported routes explicit and avoids
empty-string sentinel behavior.

Before:

```rust,ignore
impl ProviderSpec for MySpec {
    fn chat_url(&self, stream: bool, req: &ChatRequest, ctx: &ProviderContext) -> String {
        let _ = (stream, req);
        format!("{}/chat/completions", ctx.base_url.trim_end_matches('/'))
    }
}
```

After:

```rust,ignore
impl ProviderSpec for MySpec {
    fn try_chat_url(
        &self,
        stream: bool,
        req: &ChatRequest,
        ctx: &ProviderContext,
    ) -> Result<String, LlmError> {
        let _ = (stream, req);
        Ok(format!("{}/chat/completions", ctx.base_url.trim_end_matches('/')))
    }
}
```

For unsupported custom routes, leave the default `try_*_url(...)` method in place or return an
`LlmError::UnsupportedOperation` yourself when you need a provider-specific message.

## 11) `LlmError` runtime policy helpers

`LlmError` itself remains the shared error data type, but retry and presentation policy no longer
live in `siumai-spec`. This keeps the spec crate passive and moves runtime decisions to core.

Before:

```rust,ignore
use siumai_spec::error::LlmError;

let error = LlmError::api_error(500, "server");
assert!(error.is_retryable());
```

After:

```rust,ignore
use siumai_core::error::{LlmError, LlmErrorExt};

let error = LlmError::api_error(500, "server");
assert!(error.is_retryable());
```

Facade users can import the same extension trait from the unified prelude:

```rust,ignore
use siumai::prelude::unified::{LlmError, LlmErrorExt};

let error = LlmError::RateLimitError("too many requests".into());
assert!(error.is_retryable());
```

## 12) Execution middleware imports

Execution middleware is an advanced integration API rather than a stable model-family surface. If
you previously imported `LanguageModelMiddleware` from `siumai::prelude::unified::*`, import it from
the explicit experimental execution path instead.

Before:

```rust,ignore
use siumai::prelude::unified::LanguageModelMiddleware;
```

After:

```rust,ignore
use siumai::experimental::execution::middleware::LanguageModelMiddleware;
```

`MiddlewareBuilder`, `NamedMiddleware`, and preset middleware helpers also live under
`siumai::experimental::execution::middleware`.

`ReasoningTagPresets::for_model(...)` no longer routes concrete provider/model names inside
`siumai-core`. It now returns the provider-agnostic default `<think>` tag config. If your integration
needs a provider-specific tag such as `<thought>` or `<reasoning>`, choose it explicitly:

```rust,ignore
use siumai::experimental::execution::middleware::presets::{
    ExtractReasoningMiddleware,
    ReasoningTagPresets,
};

let middleware = ExtractReasoningMiddleware::with_tag(ReasoningTagPresets::thought());
```

Provider-specific default routing belongs in provider/facade extension code, not in
provider-agnostic core middleware.

`SystemMessageModeWarningMiddleware::new()` now takes the provider option namespace that should be
checked for `systemMessageMode`. Automatic middleware construction passes its configured provider
namespace, but manual middleware setup must provide it explicitly.

Before:

```rust,ignore
use siumai::experimental::execution::middleware::presets::SystemMessageModeWarningMiddleware;

let middleware = SystemMessageModeWarningMiddleware::new();
```

After:

```rust,ignore
use siumai::experimental::execution::middleware::presets::SystemMessageModeWarningMiddleware;

let middleware = SystemMessageModeWarningMiddleware::new("provider-id");
```

If a provider needs aliases or multiple provider option namespaces, that policy belongs in
provider/facade wiring rather than in `siumai-core`.

## 13) Deprecated AI SDK parity aliases

Deprecated AI SDK parity spellings are no longer part of the stable unified prelude. Use the stable
names for new code:

- `LanguageModelCallOptions` plus `RequestOptions` instead of `CallSettings`
- `GenerateImageResult` instead of `Experimental_GenerateImageResult`
- `GeneratedImage` instead of `Experimental_GeneratedImage`
- `LanguageModelStreamPart` instead of `ExperimentalLanguageModelStreamPart` or
  `Experimental_LanguageModelStreamPart`
- `SpeechResult` instead of `Experimental_SpeechResult`
- `TranscriptionResult` instead of `Experimental_TranscriptionResult`
- `filter_active_tools` instead of `experimental_filter_active_tools`
- `is_step_count` instead of `step_count_is`

When you need source-compatible migration imports, use the explicit compatibility surface:

```rust,ignore
use siumai::compat::{
    Experimental_GenerateImageResult,
    experimental_filter_active_tools,
    step_count_is,
};
```

`siumai::prelude::compat::*` provides the same names for migration modules that already use the
compat prelude.

## 14) Streaming tool-call helper imports

The low-level `StreamingToolCall*` helpers model indexed provider streaming deltas. They are useful
for compatibility and provider-utils style migrations, but they are not part of the stable facade
root or unified prelude.

Before:

```rust,ignore
use siumai::{StreamingToolCallDelta, StreamingToolCallTracker};
```

After:

```rust,ignore
use siumai::compat::{
    StreamingToolCallDelta,
    StreamingToolCallTracker,
};
```

Use semantic model-family stream types such as `ChatStreamEvent` and `ChatStreamPart` for normal
application streaming code.

## 15) Provider builder entry imports

The provider-specific builder entry is explicit compatibility surface only. The historical root
alias and root provider shim were removed so builder-style construction does not look like a stable
facade root API.

Before:

```rust,ignore
// Old code used the removed root Provider alias.
let client = Provider::openai()
    .api_key("test-key")
    .model("gpt-4o-mini")
    .build()
    .await?;
```

After:

```rust,ignore
use siumai::compat::Provider;

let client = Provider::openai()
    .api_key("test-key")
    .model("gpt-4o-mini")
    .build()
    .await?;
```

If older code imported `Siumai` or `SiumaiBuilder` from the removed root provider shim, import them
from the explicit compatibility surface instead.

Before:

```rust,ignore
use siumai::provider::{Siumai, SiumaiBuilder};
```

After:

```rust,ignore
use siumai::compat::{Siumai, SiumaiBuilder};
```

If older custom-provider code imported builder base internals from the removed root builder shim,
use the explicit compatibility builder module.

Before:

```rust,ignore
use siumai::builder::BuilderBase;
```

After:

```rust,ignore
use siumai::compat::builder::BuilderBase;
```

New code should prefer registry model handles or provider config/client constructors; this path is
for migration code that still needs method-style provider builders.

## 16) Root broad type namespace imports

The historical root `siumai::types::*` catch-all namespace moved to the explicit compatibility
surface. This keeps broad migration imports visible and prevents the facade root from becoming the
default type namespace again.

Before:

```rust,ignore
use siumai::types::{ChatMessage, Tool, Warning};
```

After, for migration code that intentionally keeps the catch-all namespace:

```rust,ignore
use siumai::compat::types::{ChatMessage, Tool, Warning};
```

Or through the compatibility prelude:

```rust,ignore
use siumai::prelude::compat::types::{ChatMessage, Tool, Warning};
```

For new code, prefer the owning stable paths instead:

```rust,ignore
use siumai::prelude::unified::{ChatMessage, Tool, Warning};
use siumai::extensions::types::ImageEditInput;
```

## 17) Provider-specific note: Vertex Gemini image

Vertex Gemini image requests now reject mask and multi-image count settings on the Gemini image
path when those settings are unsupported. This is intentional: it prevents a request from silently
being routed through the wrong provider mode.

If you need mask/reference-image behavior, use the Vertex Imagen edit path and its provider options
instead of the Gemini image generation path.

## 18) Live provider smoke tests

The live smoke script still skips missing API keys. In `0.11.0-beta.7`, Gemini defaults to
`gemini-2.5-flash-lite` for a more stable low-cost smoke path, and transient provider/network
errors are retried.

```powershell
$env:SIUMAI_TEST_PROXY = "http://127.0.0.1:10809"
$env:SIUMAI_ENV_SMOKE_PROFILE = "all-providers"
scripts\test-env-smoke.bat
```

Set `SIUMAI_ENV_SMOKE_STRICT=1` when CI should fail instead of self-skipping known account, region,
or quota denials.
