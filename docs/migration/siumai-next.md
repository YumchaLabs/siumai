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
- `google` now means the experimental configured Gemini Interactions image provider only. The
  deprecated Imagen `:predict` path and legacy Gemini language, Live, files, generated-content
  clients, and broad Vertex media surfaces are gone.
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
- The generic OpenAI-compatible engine retains explicit custom endpoints and verified ARK and Kimi
  profiles. Unverified GLM, Qianfan, Hunyuan, SiliconFlow, DeepInfra, and other named presets are no
  longer shipped as support claims.

The old `siumai-spec` and `siumai-provider-utils` packages are removed. Legacy universal client
traits, provider capability switches, provider-specific feature flags in core, compatibility
builders, completion aliases, and duplicate request/response type systems do not have replacement
aliases. Migrate directly to configured providers, family models, typed options, and typed
annotations.

The former Google `gcp` credential helper is also removed. Supply a short-lived access token with
`GoogleVertexCredential::access_token`, or implement `GoogleVertexTokenSource` in the host so token
refresh remains under the application's credential policy.

## Migration checklist

- Replace `MinimaxConfig` and `MinimaxClient` with `MinimaxCredential` and `MinimaxProvider`.
- Construct a family model explicitly and keep the configured provider long-lived.
- Replace compatibility chat request types with `LanguageRequest`, `Message`, and `MessagePart`.
- Move provider call controls into the matching typed MiniMax options and `CallOptions`.
- Move prompt-cache intent onto typed message, content, or tool annotations.
- Acquire files, image, video, music, and speech APIs from `MinimaxProvider`.
- Add Registry only for explicit local routing, and register each non-default API mode separately.
- Replace provider-wide `scope()` or `platform()` queries with `provider_id()` or an exact model or
  registration family scope.
- Keep account, region, availability, pricing, and fallback policy in the host application.
- Replace removed provider/features with an explicitly supported slice or a generic compatible
  endpoint only when protocol compatibility is sufficient for the application.

See [`../providers/minimax.md`](../providers/minimax.md) for dated support evidence and deliberate
limitations.
