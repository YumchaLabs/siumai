# Migrating to the Siumai Next API

This guide targets the breaking Siumai Next API represented by the current `0.11.0-beta.9`
workspace. It describes public API migration, not internal refactor history.

## Architectural change

The old surface centered on broad clients, capability discovery, global construction shortcuts,
and provider-option maps. The new surface has two complementary entry points:

1. provider-owned APIs for construction, protocol selection, typed options, annotations, metadata,
   and native resources;
2. provider-neutral model-family traits for portable execution.

Registry and runtime are optional integrations. Registry routes providers that the host has already
configured; it does not construct hidden providers, discover remote inventories, or choose business
routes.

The base `Provider` trait now exposes canonical provider identity only. Exact platform, protocol,
and API mode are properties of a concrete model or family registration rather than provider-wide
metadata.

## Dependency features

The facade no longer enables an AI provider by default. Select providers explicitly:

```toml
[dependencies]
siumai = { version = "0.11.0-beta.9", default-features = false, features = ["minimax"] }
tokio = { version = "1", features = ["rt-multi-thread", "macros"] }
```

Add `registry` or `runtime` only when the application uses those layers.

## OpenAI Responses module path

The temporary `siumai_protocol_openai::responses_next` module was renamed to
`siumai_protocol_openai::responses`. The old module is not retained as an alias. Update protocol
imports directly; the `openai-responses` Cargo feature and all wire identifiers remain unchanged.

## MiniMax construction

The compatibility-era `MinimaxConfig` and `MinimaxClient` path is replaced by one long-lived,
model-independent provider.

Before:

```rust,ignore
let config = MinimaxConfig::new(api_key).with_model("MiniMax-M3");
let client = MinimaxClient::from_config(config)?;
```

After:

```rust,no_run
use siumai::providers::minimax::{MinimaxCredential, MinimaxProvider};

# let api_key = String::new();
let provider = MinimaxProvider::builder(MinimaxCredential::api_key(api_key)).build()?;
let model = provider.language("MiniMax-M3")?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

Provider construction and model-handle construction are synchronous and network-free. Model IDs are
open, so private deployments and safe future IDs do not need a library release before they can be
constructed.

## Requests and execution

Replace compatibility `ChatRequest` calls and broad client methods with `LanguageRequest`, a
concrete language model, and the family helper or `LanguageModel` trait:

```rust,no_run
use siumai::families::language;
use siumai::providers::minimax::{MinimaxCredential, MinimaxProvider};
use siumai::{LanguageRequest, Message, MessageRole};

# async fn run() -> Result<(), Box<dyn std::error::Error>> {
let provider = MinimaxProvider::builder(MinimaxCredential::api_key("test-key")).build()?;
let model = provider.language("MiniMax-M3")?;
let response = language::generate(
    &model,
    LanguageRequest::new(vec![Message::text(MessageRole::User, "Hello")]),
)
.await?;
println!("{:?}", response.content());
# Ok(())
# }
```

Shared requests contain portable semantics. Provider-specific call behavior moves into typed
provider options carried by `CallOptions`:

```rust,no_run
use siumai::providers::minimax::{
    MinimaxMessagesOptions, MinimaxServiceTier, MinimaxThinking,
};
use siumai::CallOptions;

let minimax = MinimaxMessagesOptions::new()
    .with_thinking(MinimaxThinking::Adaptive)
    .with_service_tier(MinimaxServiceTier::Priority)
    .provider_options()?;
let options = CallOptions::default().with_provider_options(minimax);
# Ok::<(), Box<dyn std::error::Error>>(())
```

Do not move credentials, endpoints, authorization headers, or transport policy into provider
options. The provider builder owns those settings.

## Stream failures and terminal parity

An established language stream now settles exactly once through `StreamTerminal`. Provider errors
delivered over a successful HTTP SSE response appear as `StreamTerminal::Failed { error, .. }` with
the canonical `Error` contract. Applications should match `ErrorKind` and inspect bounded
`ResponseDiagnostics`; they should not parse provider messages or arbitrary error JSON. Exact
context-window and temporary-unavailability signals use the new
`ErrorKind::ContextWindowExceeded` and `ErrorKind::Unavailable` variants. Raw provider envelopes are
available only through the explicitly sensitive error accessor.

`SafeResponseHeaders`, `DiagnosticHeaderError`, `ResponseDiagnostics::headers`, and
`ResponseDiagnostics::with_headers` were removed. Header names cannot prove that provider-controlled
values are safe for default logs or serialization. Use the typed status, request ID, retry delay,
provider code/type/parameter, and truncation fields instead; raw headers remain available only from
the explicitly sensitive response accessor.

Chat Completions retains a trailing usage-only chunk before publishing its terminal response.
Responses streams compare an executable item shared by stable and terminal views using canonical
JSON semantics, and reject changes to its call ID, name, caller, or tool kind. Consumers no longer
need to normalize encoded tool-argument strings or reconcile disagreeing executable snapshots.

Custom compatibility decoders that wrap another `LanguageStreamDecoder` should forward
`set_response_diagnostics` to the inner decoder. The method has a default implementation for
source compatibility, but forwarding is required to retain the validated request ID and
`Retry-After` hint on in-band failures.

## Canonical messages and tool calls

The portable language boundary now validates request direction and tool execution ownership. Prefer
role-safe constructors instead of assembling arbitrary role/content pairs:

```rust,no_run
use serde_json::json;
use siumai::{ContentPart, Message, ToolCall};

let user = Message::user("Find the current record");
let assistant = Message::assistant_parts([ContentPart::ToolCall(ToolCall::local(
    "call_1",
    "lookup",
    json!({"id": 42}),
)?)])?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

`ToolCall` fields are private. Use `id()`, `name()`, `input()`, `arguments()`, and `owner()` when
reading a call. `ToolCall::local` accepts one parsed JSON value and rejects invalid identity or
oversized input. Provider-hosted programs, custom-text calls, MCP operations, computer actions, and
other provider-executed operations are provider-native output or `ProviderOpaque` replay items; they
no longer appear as portable executable tool calls. A structured function call issued by an OpenAI
hosted program is still caller-executed; Siumai pairs it with native metadata so its `caller` link is
restored when the tool result is replayed.

`LanguageRequest::validate()` rejects response-only citation/refusal content, misplaced tool
results, and other invalid role/content combinations before encoding. Code that intentionally
constructs parts dynamically must handle this typed validation result rather than relying on a
provider codec to ignore unsupported content.

## Response-to-history projection

Do not copy every `LanguageResponse::content()` part into an assistant request message. Project it
through the canonical direction boundary:

```rust,ignore
let projection = response.project_assistant_history();
for omission in projection.omissions() {
    record_projection_omission(omission);
}
if let Some(assistant) = projection.into_message() {
    history.push(assistant);
}
```

The projection preserves replayable assistant content and reports response-only citations,
refusals, and tool results as structured omissions. Siumai runtime uses this path automatically and
stores omissions in each `StepRecord`.

## Provider replay domains

Provider-native items now carry checked provenance tied to a non-secret `ReplayDomain`. Exact
provider, platform, protocol, API mode, audience, and caller scope must match before opaque history
can be replayed. Missing identity fails closed.

Official provider endpoints usually supply a stable provider-owned audience. A custom endpoint must
declare its own caller-owned audience:

```rust,ignore
use siumai::{ReplayDomain, ReplayDomainId};

let domain = ReplayDomain::custom(ReplayDomainId::new("production-relay")?)
    .with_caller_scope(ReplayDomainId::new("tenant-a")?);
let provider = provider_builder
    .with_endpoint(custom_endpoint)
    .with_replay_domain(domain)
    .build()?;
```

Use stable labels, not URLs, hostnames, API keys, signed values, or private account data. Callers
that configure multiple accounts, projects, workspaces, or deployments on one official audience
must give each replay boundary a distinct non-secret caller scope.

Anthropic on Vertex requires this boundary explicitly because the project is material technical
addressing data but must not be serialized into durable history automatically:

```rust,ignore
use siumai::providers::google_vertex_anthropic::{
    GOOGLE_VERTEX_ANTHROPIC_REPLAY_AUDIENCE, GoogleVertexAnthropicProvider,
};
use siumai::{ReplayDomain, ReplayDomainId};

let replay = ReplayDomain::official(ReplayDomainId::new(
    GOOGLE_VERTEX_ANTHROPIC_REPLAY_AUDIENCE,
)?)
.with_caller_scope(ReplayDomainId::new("vertex-project-a")?);
let provider = GoogleVertexAnthropicProvider::builder(project, location, credential)
    .with_replay_domain(replay)
    .build()?;
```

The caller-supplied label need not equal the raw project ID. It exists only to prevent replay across
materially different configured audiences.

## Node-scoped prompt caching

Prompt-cache intent no longer uses numeric message/content selectors or an untyped recursive map.
Attach a typed MiniMax cache marker to the semantic node it modifies:

```rust,no_run
use siumai::core::MessagePart;
use siumai::providers::minimax::MinimaxContentCache;
use siumai::{LanguageRequest, Message, MessageRole};

let cached_context = MessagePart::text("A large reusable context")
    .with_provider_annotation(&MinimaxContentCache::new())?;
let request = LanguageRequest::new(vec![Message::new(
    MessageRole::User,
    [cached_context],
)]);
# Ok::<(), Box<dyn std::error::Error>>(())
```

Message- and tool-level equivalents are `MinimaxMessageCache` and `MinimaxToolCache`. These markers
apply only to the Messages API mode and encode MiniMax's fixed ephemeral cache control.

## Explicit MiniMax language modes

`provider.language(model)` now has a documented meaning: it selects the recommended
Anthropic-compatible Messages mode.

| Requirement | Constructor | Matching typed options |
|---|---|---|
| Recommended Messages mode | `provider.language(model)` or `provider.messages(model)` | `MinimaxMessagesOptions` |
| OpenAI Chat Completions wire API | `provider.chat_completions(model)` | `MinimaxChatCompletionsOptions` |
| Bounded OpenAI Responses subset | `provider.responses(model)` | `MinimaxResponsesOptions` |

Options are mode-specific. Passing typed options for another mode fails closed instead of being
silently ignored.

## MiniMax native resources

Provider-specific resources no longer hang from a universal capability client. Acquire them from
the configured provider:

```rust,no_run
# use siumai::providers::minimax::{MinimaxCredential, MinimaxProvider};
# let provider = MinimaxProvider::builder(MinimaxCredential::api_key("test-key")).build()?;
let files = provider.files();
let images = provider.images();
let video = provider.video();
let music = provider.music();
let speech = provider.speech();
# Ok::<(), Box<dyn std::error::Error>>(())
```

Each resource has typed request, response, identifier, status, and validation types under
`siumai::providers::minimax::resources`. Video and asynchronous speech expose explicit submit/query
operations; they do not start a hidden polling loop. Music and synchronous HTTP speech currently
return buffered non-streaming responses.

## Provider-owned Alibaba video jobs

The unimplemented `StreamingTranscriptionModel` and the one-provider `VideoJobModel` and `MediaJob`
with its untyped `Value` state were removed from `siumai-core::experimental`. Asynchronous media
lifecycles remain provider-owned until multiple implementations demonstrate genuinely portable
semantics.

Alibaba callers should use `AlibabaVideoModel::create`, `poll`, `cancel`, and `materialize` with
`AlibabaVideoRequest` and `AlibabaVideoJob`. The typed job now owns its
validated `AlibabaVideoJobId` and `AlibabaVideoJobStatus` and is directly serializable. Its signed
download URL is never persisted; materializing a restored completed job polls once to obtain a
fresh URL. Snapshot diagnostic fields are bounded and control-free; use
`AlibabaVideoUsage::size()` for the validated output-size text. No compatibility aliases are
retained for the removed core types.

## Optional Registry migration

There is no global provider registry populated from environment variables. Build an immutable
Registry from explicit provider registrations:

```rust,no_run
use siumai::providers::minimax::{MinimaxCredential, MinimaxProvider};
use siumai::registry::{Registry, RegistryBuilderExt};

let provider = MinimaxProvider::builder(MinimaxCredential::api_key("test-key")).build()?;
let mut builder = Registry::builder();
builder.register_provider("minimax", &provider)?;
let registry = builder.build()?;
let model = registry.language_model("minimax:MiniMax-M3")?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

Custom providers now build non-empty registrations from a first family and add only disjoint
families:

```rust,ignore
let registration = ProviderRegistration::from_language(language_scope, language_policy, language)
    .bind_transcription(transcription_scope, transcription_policy, transcription)?;
```

The removed `ProviderRegistration::new(scope, policy).with_*` shape stored one provider-wide scope
and policy. Query the replacement with an explicit family, for example
`registration.api_mode(ModelFamily::Language)`. Same-family alternate API modes remain separate
registrations and should be assigned distinct Registry routes when both are needed.

`ModelOperation` now determines the family for policy evaluation, so calls use
`registration.evaluate(model, ModelOperation::Embed)` rather than passing a second family value.
Use `registration.for_family(ModelFamily::Rerank)` when a combined provider registration should be
narrowed before assigning a route.

Facade `register_provider` is fallible at both seams. It reports normal Registry build errors and
`RegisterProviderError::NoPortableFamilyRegistration` when a provider is validly configured only for
provider-native resources or jobs. It never creates an empty registration.

The standard MiniMax registration resolves the Messages mode. Register
`provider.chat_completions_registration()` or `provider.responses_registration()` explicitly under
separate host-owned route names when those modes are required.

## Region and availability policy

Remove application dependencies on SDK-owned region/model availability enums or static commercial
catalogs. Siumai accepts technical endpoint inputs needed to address a request, but the host owns:

- account, project, workspace, deployment, and region selection;
- aliases and default models;
- current entitlement and commercial availability;
- pricing, quota, compliance, health, weights, and fallback.

This boundary keeps provider execution deterministic without pretending that a released Rust crate
can model mutable account-specific availability.

## Removed or narrowed provider surfaces

The reset removes public packages and features that did not satisfy the configured-provider,
typed-extension, shared-transport, and offline-fixture contracts. Do not treat removal as an alias
or a temporary hidden feature:

- Azure OpenAI, Amazon Bedrock, AI Gateway, Ollama, and Together AI are no longer in the workspace,
  facade, or release feature graph. They can return only as clean provider implementations.
- `google` now owns a product-level `GeminiProvider`; it no longer exposes an image-shaped provider
  identity. The current portable slices are stable-v1 Interactions language and image generation.
  The deprecated Imagen `:predict` path and the old broad Gemini/Vertex clients remain removed.
- Anthropic Messages on Vertex is available separately through the
  `google-vertex-anthropic` feature and `providers::google_vertex_anthropic`. Its project and
  location inputs are technical addressing data, not availability metadata.
- xAI retains Responses and Chat Completions language modes. Its old files, image, speech, video,
  and generic native-runtime surfaces are removed.
- Cohere retains embeddings and reranking; the old chat surface is removed.
- Cohere model hints now track Embed v4 and Rerank v4/v3. Obsolete Embed v2 hints are removed;
  textual future or legacy IDs remain callable through the open model-ID contract.
- Deepgram publishes current Nova-3/Nova-2 prerecorded hints only. Legacy Base, Enhanced, and Nova
  constants are removed without turning the remaining constants into an allowlist.
- ElevenLabs retains speech synthesis; the old transcription and broad resource clients are
  removed.
- OpenAI retains configured Chat Completions, Responses, Responses resources, and opt-in Realtime.
  The old universal client and unrelated files, image, audio, moderation, rerank, and skills
  surfaces are removed.
- The generic OpenAI-compatible engine retains explicit generic and custom endpoints only. Kimi is
  owned by `siumai-provider-moonshotai`; Volcengine ARK is owned by
  `siumai-provider-volcengine`. Their typed options, model advisories, and support evidence no
  longer appear below the compatibility-engine namespace. Unverified GLM, Qianfan, Hunyuan,
  SiliconFlow, DeepInfra, and other named presets are not shipped as support claims.

The old `siumai-spec` and `siumai-provider-utils` packages are removed. Legacy universal client
traits, provider capability switches, provider-specific feature flags in core, compatibility
builders, completion aliases, and duplicate request/response type systems do not have replacement
aliases. Migrate directly to configured providers, family models, typed options, and typed
annotations.

The Gemini image migration is intentionally alias-free:

| Removed | Replacement |
|---|---|
| `GoogleImageProvider` | `GeminiProvider` |
| `GoogleImageProviderBuilder` | `GeminiProviderBuilder` |
| `GoogleCredential` | `GeminiCredential` |
| `GoogleImageModel` | `GeminiImageModel` |
| `GoogleImageOptions` | `GeminiImageOptions` |
| `GoogleImageAspectRatio` | `GeminiImageAspectRatio` |
| `GoogleImageSize` | `GeminiImageSize` |
| `current_models()` | `current_image_models()` |

Gemini Interactions wire types now belong to `siumai-protocol-gemini`. Requests use the stable
`v1/interactions` target and the current polymorphic `response_format` object. Caller-controlled
Gemini endpoints require an explicit custom `ReplayDomain` and cannot inherit Google's verified
support evidence.

`GeminiProvider::language(model)` now selects stable-v1 Interactions and is equivalent to the
explicit `GeminiProvider::interactions(model)` entry point. Language-specific controls use
`GeminiInteractionsOptions`; storage is disabled unless explicitly enabled. The provider
registration now binds the portable Language, Embedding, Image, and Speech families.

Stable-v1 Generate Content remains available only through the explicit
`GeminiProvider::generate_content(model)` handle or `generate_content_registration()`. It does not
replace the primary Interactions route. Google currently labels this product path Legacy even
though the stable-v1 REST operations remain published; Siumai records those lifecycle facts
separately.

Additional product-level entry points are now provider-owned:

- `embedding(model)` implements the stable-v1 text embedding subset;
- `speech(model)` implements buffered Interactions TTS and requires an explicit voice;
- `files()` implements stable-v1 metadata get/list/delete, without upload or GCS registration;
- `veo()` implements typed Veo submit/status without hidden polling or automatic download.

Veo remains a provider-native job API. The removed generic `VideoJobModel` and `MediaJob<Value>`
types are not reintroduced.

The former Google `gcp` credential helper is also removed. Supply a short-lived access token with
`GoogleVertexCredential::access_token`, or implement `GoogleVertexTokenSource` in the host so token
refresh remains under the application's credential policy.

Runtime durable snapshots now use schema version 5. Earlier development snapshots lack the checked
execution scope and assistant-history omission records required by this boundary and are not
migrated automatically. Recreate them from trusted application history instead of synthesizing
provider provenance.

## Migration checklist

- Replace `MinimaxConfig` and `MinimaxClient` with `MinimaxCredential` and `MinimaxProvider`.
- Construct a family model explicitly and keep the configured provider long-lived.
- Replace compatibility chat request types with `LanguageRequest`, `Message`, and `MessagePart`.
- Replace direct `ToolCall` field construction with `ToolCall::local` and checked accessors.
- Use role-safe message constructors and project responses with
  `LanguageResponse::project_assistant_history()` before appending assistant history.
- Move provider call controls into the matching typed MiniMax options and `CallOptions`.
- Move prompt-cache intent onto typed message, content, or tool annotations.
- Acquire files, image, video, music, and speech APIs from `MinimaxProvider`.
- Add Registry only for explicit local routing, and register each non-default API mode separately.
- Replace provider-wide `scope()` or `platform()` queries with `provider_id()` or an exact model or
  registration family scope.
- Keep account, region, availability, pricing, and fallback policy in the host application.
- Declare an explicit custom replay domain for every custom language endpoint, and separate material
  accounts, projects, workspaces, or deployments with non-secret caller scopes.
- Recreate pre-version-5 runtime snapshots from trusted application history.
- Replace removed provider/features with an explicitly supported slice or a generic compatible
  endpoint only when protocol compatibility is sufficient for the application.

See [`../providers/minimax.md`](../providers/minimax.md) for dated support evidence and deliberate
limitations.
