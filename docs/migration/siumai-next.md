# Migrating to the Siumai Next API

This guide targets the breaking Siumai Next API in the current repository. The focused facade
section below starts from the published `0.11.0-beta.10` surface; the remaining sections describe
the broader Next migration. This guide documents public API migration, not internal refactor
history.

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

## Facade convergence after `0.11.0-beta.10`

The facade now has one canonical portability path: choose or resolve a model, call the matching root
family module, attach typed provider intent through that call when needed, and keep the concrete
provider for native APIs. The `siumai::families` umbrella, suffixed facade option helpers, flat
`*Call` re-exports, runtime free single-call helpers, facade relays for those helpers, and the
runtime-owned `AgentInput` adapter are removed rather than kept as aliases.

### Exact symbol map

Default family calls keep their operation names and move from `siumai::families` to root modules:

| `0.11.0-beta.10` symbol | Current replacement | Behavioral note |
|---|---|---|
| `use siumai::families::{language, embedding, rerank, image, speech, transcription}` | `use siumai::{language, embedding, rerank, image, speech, transcription}` | The root modules are the only canonical facade family namespace. |
| `siumai::families::language::generate` | `siumai::language::generate` | Also accepts `&str`, `String`, one `Message`, a message list, or a complete `LanguageRequest` through `LanguageInput`; returns the complete `LanguageResponse`. |
| `siumai::families::language::stream` | `siumai::language::stream` | Accepts the same language input forms and returns the existing established `LanguageStream`. |
| `siumai::families::embedding::embed` | `siumai::embedding::embed` | Keeps the complete `EmbeddingRequest` and `EmbeddingResponse`. |
| `siumai::families::rerank::rerank` | `siumai::rerank::rerank` | Keeps the complete query/candidate request and response. |
| `siumai::families::image::generate` | `siumai::image::generate` | Keeps the complete `ImageRequest` and `ImageResponse`. |
| `siumai::families::speech::synthesize` | `siumai::speech::synthesize` | Keeps the complete `SpeechRequest` and buffered response. |
| `siumai::families::transcription::transcribe` | `siumai::transcription::transcribe` | Keeps the complete audio request and final-result response. |

The seven facade helpers that accepted `CallOptions` as a separate argument move to bound calls:

| Removed `0.11.0-beta.10` helper | Current replacement | Behavioral note |
|---|---|---|
| `siumai::families::language::generate_with_options(&model, input, options)` | `siumai::language::call(&model, input).with_options(options)?.generate().await` | `with_options` validates the complete candidate synchronously; generation still returns the complete response. |
| `siumai::families::language::stream_with_options(&model, input, options)` | `siumai::language::call(&model, input).with_options(options)?.stream().await` | Stream establishment and terminal semantics are unchanged. |
| `siumai::families::embedding::embed_with_options(&model, request, options)` | `siumai::embedding::call(&model, request).with_options(options)?.embed().await` | No universal input or hidden defaults are added. |
| `siumai::families::rerank::rerank_with_options(&model, request, options)` | `siumai::rerank::call(&model, request).with_options(options)?.rerank().await` | Query and candidates remain explicit. |
| `siumai::families::image::generate_with_options(&model, request, options)` | `siumai::image::call(&model, request).with_options(options)?.generate().await` | Returns the complete image response. |
| `siumai::families::speech::synthesize_with_options(&model, request, options)` | `siumai::speech::call(&model, request).with_options(options)?.synthesize().await` | Returns the complete buffered speech response. |
| `siumai::families::transcription::transcribe_with_options(&model, request, options)` | `siumai::transcription::call(&model, request).with_options(options)?.transcribe().await` | Returns the complete transcription response. |

These replacements have two explicit error phases. `with_options` and
`with_provider_options` validate the complete candidate synchronously and return
`ProviderOptionError`; no model call has been dispatched when either method fails. The terminal
language `generate` method still returns `LanguageCallError`, while language `stream` and the five
non-language terminal methods retain their family `Error` result. Code that previously returned
only the terminal error type must widen its application error or map the setup error explicitly:

```rust,ignore
#[derive(Debug, thiserror::Error)]
enum ApplicationCallError {
    #[error("invalid provider options")]
    Setup(#[from] siumai::ProviderOptionError),
    #[error("language call failed")]
    Call(#[from] siumai::LanguageCallError),
}

async fn generate_with_options<M>(
    model: &M,
    request: siumai::LanguageRequest,
    options: siumai::CallOptions,
) -> Result<siumai::LanguageResponse, ApplicationCallError>
where
    M: siumai::LanguageModel + ?Sized,
{
    Ok(siumai::language::call(model, request)
        .with_options(options)?
        .generate()
        .await?)
}
```

Typed provider options normally no longer need a separately assembled `CallOptions` value:

```rust,ignore
let response = siumai::language::call(&model, "Summarize the request")
    .with_provider_options(&provider_options)?
    .generate()
    .await?;
```

The patch binds to the exact live model, including configured-instance identity and canonical
Registry route. Typed annotations remain on the message, content part, or tool they modify. If an
application already owns a complete `CallOptions` baseline, pass it through `with_options` and add
typed patches through subsequent `with_provider_options` calls; their order is preserved.

Call-builder types remain public in their owning family modules, but are no longer flattened into
the facade root or prelude:

| Removed flat symbol | Current replacement |
|---|---|
| `siumai::LanguageCall` or `LanguageCall` from `siumai::prelude::*` | `siumai::language::LanguageCall` |
| `siumai::EmbeddingCall` or `EmbeddingCall` from `siumai::prelude::*` | `siumai::embedding::EmbeddingCall` |
| `siumai::RerankCall` or `RerankCall` from `siumai::prelude::*` | `siumai::rerank::RerankCall` |
| `siumai::ImageCall` or `ImageCall` from `siumai::prelude::*` | `siumai::image::ImageCall` |
| `siumai::SpeechCall` or `SpeechCall` from `siumai::prelude::*` | `siumai::speech::SpeechCall` |
| `siumai::TranscriptionCall` or `TranscriptionCall` from `siumai::prelude::*` | `siumai::transcription::TranscriptionCall` |

Runtime callers now name the configured runtime explicitly. Pass `StepOptions::default()` to keep
the former helper behavior:

| Removed runtime helper | Current replacement |
|---|---|
| `siumai::generate`, `generate` from `siumai::prelude::*`, or `siumai::runtime::generate` | `siumai::runtime::Runtime::default().generate(&model, request, siumai::runtime::StepOptions::default(), options)` |
| `siumai::stream`, `stream` from `siumai::prelude::*`, or `siumai::runtime::stream` | `siumai::runtime::Runtime::default().stream(&model, request, siumai::runtime::StepOptions::default(), options)` |
| `siumai_runtime::generate` | `siumai_runtime::Runtime::default().generate(&model, request, siumai_runtime::StepOptions::default(), options)` |
| `siumai_runtime::stream` | `siumai_runtime::Runtime::default().stream(&model, request, siumai_runtime::StepOptions::default(), options)` |

Use `siumai::language::generate` or `siumai::language::stream` instead when no runtime defaults or
step-level orchestration are required.

### `AgentInput` to `LanguageInput`

`siumai_runtime::AgentInput` is removed. Agent methods now accept `Into<LanguageInput>`, sharing the
same provider-neutral conversion as facade language calls. Strings, messages, message lists, and
complete requests continue to work without wrapping.

If downstream code implemented a custom conversion into `AgentInput`, move that conversion to
`LanguageInput`:

```rust,no_run
use siumai::{LanguageInput, LanguageRequest};

struct LocalAgentInput(LanguageRequest);

impl From<LocalAgentInput> for LanguageInput {
    fn from(input: LocalAgentInput) -> Self {
        input.0.into()
    }
}
```

Applications depending directly on the owning crates use `siumai_core::LanguageInput` with
`siumai_runtime::Agent`. The conversion immediately preserves the complete canonical request; Agent
instruction prepending, request validation, tools, structured output, annotations, and runtime
behavior are otherwise unchanged.

Direct callers of the removed `AgentInput::into_request` method should use
`LanguageInput::into_request`.

### Complete language results and native escape paths

The concise call still returns a complete `LanguageResponse`. Use `output_text()` only when the
application needs a display string. It concatenates canonical text parts in order and returns
`None` when there is no text part; reasoning, refusals, tools, citations, media, provider-native
state, termination, usage, warnings, and metadata remain on the response. It does not replace
`project_assistant_history()` for continuation.

A Registry-resolved model exposes the same family trait accepted by the root module, so portable
application code does not match on providers. Registry does not downcast that trait object into
native capabilities. Retain the concrete configured provider beside Registry when the application
also needs provider-owned files, batches, catalogs, sessions, hosted tools, or media jobs.

## Breaking ownership API map

The following changes are intentional removals, not compatibility aliases:

| Former surface | Current replacement | New owner |
|---|---|---|
| `ModelPolicy`, `SupportState`, `UnsupportedReason`, and provider/Registry `evaluate` calls | Optional host preflight over a provider-owned profile or support manifest | Host application |
| `ProviderRegistration::new(scope, policy).with_*` | `ProviderRegistration::from_*(scope, factory)` plus `bind_*(scope, factory)` | Registry/core registration |
| Public `ProviderOptionOrigin`, `ProviderOptionLayers`, `ProviderOptionContext`, and `ProviderOptionMerger` | `CallOptions::with_provider_options_for(&model, &typed)` with runtime-private ordered assembly | Core + runtime + provider |
| Erased `CallOptions::with_provider_options(ProviderOptions)` | Typed `with_provider_options(&typed)` only for explicitly reusable values; otherwise exact-target insertion | Core |
| Namespace-only raw body insertion | `with_raw_provider_json_for(&model, bytes)` or exact-target `Value` insertion | Core + provider codec |
| `LanguageResponseStatus` × `FinishReason` | `LanguageTermination::{Completed, Incomplete}` | Core language contract |
| Failed/cancelled terminal `response: Option<LanguageResponse>` | `LanguageCallError::partial()` or `StreamTerminal` partial output | Core language/runtime |
| `LanguageStreamEvent::Usage(Usage)` | `LanguageStreamEvent::Usage(UsageUpdate)` with `Snapshot`/`Delta` | Core stream/runtime |
| Lifecycle warning kinds (`UnknownModel`, `DeprecatedModel`, `RetiredModel`, `RollingModelAlias`) | Explicit support-profile inspection before host routing | Host/provider introspection |

If a migration needs a provider field that this release does not yet model, use the provider's
typed options or its reviewed raw-body escape hatch. Do not reintroduce a global field denylist or a
model-name capability gate merely to preserve an old call site.

## Dependency features

The facade no longer enables an AI provider by default. Select providers explicitly:

```toml
[dependencies]
siumai = { version = "0.11.0-beta.10", default-features = false, features = ["minimax"] }
tokio = { version = "1", features = ["rt-multi-thread", "macros"] }
```

Add `registry` or `runtime` only when the application uses those layers.

OpenAI Responses WebSocket is an independent provider-native feature. It enables the base OpenAI
provider but does not enable Realtime:

```toml
[dependencies]
siumai = { version = "0.11.0-beta.10", default-features = false, features = ["openai-responses-websocket"] }
```

## Provider HTTP settings, trusted CONNECT routes, call deadlines, and retry caps

Every configured provider and both compatibility engines now accept one transport-owned stateless
HTTP settings value. Facade provider features enable the curated `siumai::transport` namespace, so
facade applications do not need a direct `siumai-transport` dependency:

```rust,ignore
use std::time::Duration;
use siumai::providers::openai::{OpenAiCredential, OpenAiProvider};
use siumai::transport::{ProviderHttpTransportSettings, RetryPolicy};

let settings = ProviderHttpTransportSettings::default()
    .with_retry_policy(RetryPolicy::new(2)?)
    .with_connect_timeout(Duration::from_secs(10))?
    .with_call_timeout(Duration::from_secs(120))?
    .with_read_timeout(Duration::from_secs(30))?;

let provider = OpenAiProvider::builder(OpenAiCredential::api_key("example-key"))
    .with_http_transport_settings(settings)
    .build()?;
```

To use an explicit forward proxy, construct the route separately and apply it to the same
settings value. This example only validates configuration; it does not discover a proxy or make a
network request:

```rust,ignore
use siumai::transport::{HttpTransportRoute, ProviderHttpTransportSettings, ProxyEndpoint};

let proxy = ProxyEndpoint::https("https://proxy.example.com")?;
let route = HttpTransportRoute::trusted_connect(proxy);
let settings = ProviderHttpTransportSettings::default().with_route(route)?;
// Pass `settings` to a provider builder with `with_http_transport_settings(settings)`.
```

The old-to-new migration map is intentionally alias-free:

| Removed provider-builder surface | Current replacement |
|---|---|
| `with_limits(...)` or `with_transport_limits(...)` for stateless provider HTTP | `ProviderHttpTransportSettings::with_limits(...)`, then builder `with_http_transport_settings(...)` |
| `OpenAiProviderBuilder::with_transport_limits(...)` for Realtime | `OpenAiProviderBuilder::with_realtime_transport_limits(...)` |
| `OpenAiProviderBuilder::with_transport_limits(...)` for Responses WebSocket | `OpenAiProviderBuilder::with_responses_websocket_transport_limits(...)` |
| `with_retry_policy(...)` | `ProviderHttpTransportSettings::with_retry_policy(...)` |
| `with_connect_timeout(...)` for stateless provider HTTP | `ProviderHttpTransportSettings::with_connect_timeout(...)` |
| `OpenAiProviderBuilder::with_connect_timeout(...)` for Realtime | `OpenAiProviderBuilder::with_realtime_connect_timeout(...)` |
| `OpenAiProviderBuilder::with_connect_timeout(...)` for Responses WebSocket | `OpenAiProviderBuilder::with_responses_websocket_connect_timeout(...)` |
| `with_call_timeout(...)` | `ProviderHttpTransportSettings::with_call_timeout(...)` |
| `with_read_timeout(...)` | `ProviderHttpTransportSettings::with_read_timeout(...)` |
| `OpenAiProviderBuilder::with_transport_observer(...)` | `ProviderHttpTransportSettings::with_observer(...)` |
| Provider builder proxy/client or environment configuration | `ProviderHttpTransportSettings::with_route(HttpTransportRoute::trusted_connect(...))`; there is no raw client, custom-fetch, or environment-proxy replacement |
| Proxy URL userinfo or opaque `Proxy-Authorization` header | `ProxyEndpoint` plus optional bounded `ProxyBasicCredential` for HTTPS proxy negotiation only |
| Constructing `Instant::now() + duration` for relative call intent | `siumai::CallOptions::with_timeout(duration)`; keep `with_deadline(...)` for a true absolute deadline |
| `without_retry()` when the caller needs a cap greater than one | `siumai::CallOptions::with_max_attempts(n)`; `without_retry()` remains the one-attempt shorthand |
| Direct `siumai_transport` imports in a facade-only application | `siumai::transport::{ProviderHttpTransportSettings, RetryPolicy, TransportLimits, TransportObserver, ...}` |

Endpoint policy, credentials/signing, retry classification, and operation replay proof remain
separate authorities. Provider options and request bodies cannot override them. A custom endpoint
continues to describe a reverse gateway destination and its credential/replay audience; it is not a
forward proxy.

Direct mode remains the default and validates provider DNS/peer addresses locally. Trusted CONNECT
mode validates the selected proxy endpoint/peer and transfers destination DNS/peer selection to
that proxy while retaining the logical provider URL, inner TLS hostname/certificate, provider
credential audience, redirect policy, replay proof, deadlines, and resource bounds. An optional
`ProxyBasicCredential` is sent only during CONNECT; provider authentication is sent only inside the
tunnel. The credential is immutable, so rotate it by rebuilding the configured provider.

Call-level timing and retry intent now live on `CallOptions`:

```rust,ignore
use std::time::Duration;
use siumai::CallOptions;

let options = CallOptions::default()
    .with_timeout(Duration::from_secs(30))?
    .with_max_attempts(2)?;
```

The relative timeout starts when the outer family call or runtime run accepts the options, resolves
once to an absolute deadline, and does not restart during validation, queueing, backoff, streaming
establishment, runtime steps, or structured-output repair. If `with_deadline(...)` is also present,
the earlier deadline wins. `with_max_attempts(...)` caps total attempts for each logical provider
HTTP call; it cannot make `ReplaySafety::Never` replayable. `without_retry()` remains the
one-attempt convenience.

Transport observers report structural attempt-loop data only. They end at buffered response return
or stream establishment and never receive URLs, endpoint queries, headers, credentials, response
bodies, prompts, tool payloads, provider errors, or provider identity. Applications that need
provider/route/account attribution should wrap the observer when configuring each provider.

The settings value does not configure OpenAI Realtime or Responses WebSocket sessions, Alibaba
video materialization downloads, provider-returned external resources, media jobs, or MCP. Keep
using their independently named lifecycle controls. In particular, OpenAI retains dedicated
Realtime and Responses WebSocket limits/connect/session/I/O/turn settings, while Alibaba retains
`with_video_download_limits`, `with_video_download_connect_timeout`,
`with_video_download_timeout`, and `with_video_download_read_timeout`.

Streamable HTTP MCP uses the same route type without importing provider settings or provider
authentication:

```rust,ignore
use siumai_mcp::{HttpTransportRoute, McpClientConfig, ProxyEndpoint};

let route = HttpTransportRoute::trusted_connect(ProxyEndpoint::https(
    "https://proxy.example.com",
)?);
let config = McpClientConfig::default().with_http_transport_route(route);
// Pass `config` to `McpClient::from_http` when the host is ready to connect.
```

MCP retains its own endpoint policy, bearer authentication, message bounds, session lifecycle,
and never-replay semantics. POST, GET/SSE, and DELETE/session-close requests use the selected
route. Stdio MCP is unchanged. Environment proxy discovery remains disabled.

The supported route is deliberately bounded to explicit CONNECT for HTTPS destinations. SOCKS,
PAC, named proxy-product certification, opaque or bearer/Negotiate/NTLM/Kerberos proxy
authentication, URL-userinfo credentials, raw client/custom-fetch injection, custom proxy CA or
mTLS configuration, WebSocket/Realtime proxying, provider-returned external-download proxying,
and arbitrary-URL routing remain unsupported.

## OpenAI Responses module path

The temporary `siumai_protocol_openai::responses_next` module was renamed to
`siumai_protocol_openai::responses`. The old module is not retained as an alias. Update protocol
imports directly; the `openai-responses` Cargo feature and all wire identifiers remain unchanged.

## OpenAI provider families and resources

`OpenAiProvider` is now the long-lived owner of five portable model families. Its default
`registration()` binds Language through Responses plus Embedding, Image, Speech, and
Transcription. Code that needs only one language protocol should use the explicit registration:

```rust,ignore
let all_portable_families = provider.registration();
let responses_only = provider.responses_registration();
let chat_only = provider.chat_completions_registration();
```

Model acquisition remains synchronous and network-free:

```rust,ignore
let embedding = provider.embedding("text-embedding-3-small")?;
let image = provider.image("gpt-image-2")?;
let speech = provider.speech("gpt-4o-mini-tts")?;
let transcription = provider.transcription("gpt-4o-transcribe")?;
```

The provider builder accepts typed defaults independently for each family through
`with_embedding_defaults`, `with_image_defaults`, `with_speech_defaults`, and
`with_transcription_defaults`. Buffered speech requires an explicit voice; Siumai no longer
chooses one implicitly. Explicit embedding dimensions are rejected for model families whose
official request contract does not support that override, including unknown future IDs, rather
than being silently omitted.

Provider-native lifecycle APIs remain separate from the portable families:

```rust,ignore
let conversations = provider.conversations();
let files = provider.files();
let vector_stores = provider.vector_stores();
// Import `OpenAiSkillsProviderExt` from the experimental facade namespace first.
let skills = provider.skills();
```

Each operation has a normal method using default `CallOptions` and a matching `_with_options`
variant. Binary file and skill content is returned as `OpenAiBinaryContent`, whose `Debug` output
redacts the payload. The implemented slices are deliberately narrow: basic conversation item,
file, vector-store file, and directory-skill lifecycles are present; conversation item lookup,
vector search and file batches, zip-skill upload, and a universal resource client are not claimed.

The former broad native support ID `responses-resources` is replaced by
`responses-resource-lifecycle`, with separate support claims for Conversations, Files, Vector
Stores, and Skills. Applications that persist or inspect support manifests should migrate those
IDs directly.

## OpenAI stateless execution ownership

Official OpenAI Chat Completions and Responses now share the stateless HTTP/SSE execution kernel in
`siumai-openai-compatible`. This is an internal ownership change for normal OpenAI users: provider
construction, typed options, annotations, native Responses results and frames, background work,
resources, Realtime, Responses WebSocket, and support evidence remain in
`siumai-provider-openai`.

Enabling the facade `openai` feature does not expose or activate the facade
`openai-compatible` feature. Applications that only use official OpenAI do not need to add a
compatible-provider feature or construct an `OpenAiCompatibleProvider`.

Direct provider authors may use the semver-covered
`siumai_openai_compatible::extension::v2` contract with an already selected transport, bounded
prepared body, relative target, replay safety, diagnostics context, and provider-owned decoders.
It intentionally cannot choose credentials, endpoints, retry or timeout policy, provider identity,
support claims, or branded wire semantics. Existing `extension::v1` codec policies remain
available; this release does not force branded compatible providers to rewrite them.

## OpenAI Responses WebSocket sessions

Responses WebSocket is provider-owned and experimental. It does not add a seventh portable model
family and is not part of `LanguageModel::stream`. Acquire the session from an OpenAI Responses
model:

```rust,ignore
use siumai::providers::openai::experimental::responses_websocket::OpenAiResponsesWebSocketEvent;
use siumai::{CallOptions, LanguageRequest, Message};

let model = provider.responses("gpt-5.6")?;
let session = model
    .websocket()?
    .connect(CallOptions::default())
    .await?;

let turn = session
    .generate(
        LanguageRequest::new(vec![Message::user("Continue the task")]),
        CallOptions::default(),
    )
    .await?;

// Poll `turn` as a Stream<Item = Result<OpenAiResponsesWebSocketEvent, Error>>.
```

The former `siumai_provider_openai::experimental::responses_websocket::advanced` connector,
socket, sender, and receiver exports and their
`siumai::providers::openai::experimental::responses_websocket::advanced` facade mirror were
removed. So was
`OpenAiResponsesWebSocketConfig::with_connector`. Applications should use the high-level
`OpenAiResponsesWebSocketConfig`, `OpenAiResponsesWebSocketSession`, and
`OpenAiResponsesWebSocketTurn` API shown above. Production transport ownership and deterministic
transport adapters are provider-private implementation details; there is no application-level
replacement connector seam. This removal affects Responses WebSocket only and does not change the
separate Realtime session API.

One connection accepts one active response. After a terminal event, another generated turn may use
the normal typed `previous_response_id` option. `warm_up` sends `generate: false` and produces only
native warm-up frames; it never fabricates a `LanguageResponse`. Dropping or cancelling a turn
settles that exact turn, while malformed frames, unexpected EOF, queue exhaustion, or protocol
desynchronization close the session conservatively. Retryable WebSocket availability close codes
such as service restart, try again later, and bad gateway become sanitized
`ErrorKind::Unavailable`; provider-controlled close reasons are not copied into the turn error.

Turn startup and execution now expose submission certainty through
`OpenAiResponsesWebSocketSubmissionState`. A cancellation or deadline while the payload is still
provably outside the socket sender is `NotSubmitted` and can be retried safely. Once the command
enters an uncertain actor/acknowledgement race or the socket sender is polled without a provider
terminal, the result is `Indeterminate`; callers must not replay it automatically. A turn becomes
`Settled` only after an authoritative Responses terminal event. Inspect a returned turn with
`turn.submission_state()`, or classify a startup error with
`OpenAiResponsesWebSocketSubmissionState::from_error(&error)`.

The session now owns and monitors its actor task. Actor panic or abort, socket failure, and an
unsettled turn-channel close emit exactly one typed failure followed by EOF. Queue and
acknowledgement waits share the caller cancellation and deadline, and dropping the last session or
turn handle follows the same bounded actor cleanup path. This lifecycle contract was verified
against the
[official OpenAI Responses WebSocket mode documentation](https://developers.openai.com/api/docs/guides/websocket-mode/)
on 2026-08-14.

The official provider constructor supplies the provider-owned
`wss://api.openai.com/v1/responses` endpoint and publishes the experimental
`responses-websocket` native support claim. A custom HTTP endpoint has no inferred WebSocket route:
configure one explicitly with `OpenAiProviderBuilder::with_responses_websocket_endpoint`, together
with the custom replay domain required by the HTTP provider. Caller-controlled endpoints never gain
the official claim, even if their transport policy is labelled official.

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

Replace compatibility `ChatRequest` calls and broad client methods with a concrete language model
and the root language facade. A plain prompt uses the shared `LanguageInput` conversion; construct a
complete `LanguageRequest` only when the call needs messages, tools, structured output, annotations,
or other portable request fields:

```rust,no_run
use siumai::language;
use siumai::providers::minimax::{MinimaxCredential, MinimaxProvider};

# async fn run() -> Result<(), Box<dyn std::error::Error>> {
let provider = MinimaxProvider::builder(MinimaxCredential::api_key("test-key")).build()?;
let model = provider.language("MiniMax-M3")?;
let response = language::generate(&model, "Hello").await?;
println!("{:?}", response.output_text());
# Ok(())
# }
```

Shared requests contain portable semantics. Provider-specific call behavior moves into typed
provider options bound through the same family call:

```rust,no_run
use siumai::language;
use siumai::providers::minimax::{
    MinimaxCredential, MinimaxMessagesOptions, MinimaxProvider, MinimaxServiceTier,
    MinimaxThinking,
};

# async fn run() -> Result<(), Box<dyn std::error::Error>> {
let provider = MinimaxProvider::builder(MinimaxCredential::api_key("test-key")).build()?;
let model = provider.language("MiniMax-M3")?;
let minimax = MinimaxMessagesOptions::new()
    .with_thinking(MinimaxThinking::Adaptive)
    .with_service_tier(MinimaxServiceTier::Priority);
let response = language::call(&model, "Hello")
    .with_provider_options(&minimax)?
    .generate()
    .await?;
println!("{:?}", response.output_text());
# Ok(())
# }
```

The bound call validates typed options against the exact configured model instance, family, API
mode, and selected Registry route before execution. This prevents credentials, replay-sensitive
state, and provider-body authorization from crossing two configurations that happen to share public
provider labels. Lower-level option assembly remains available through
`CallOptions::with_provider_options_for(&model, &options)`. Routing fallbacks use the
`with_optional_*_for` methods. Raw forward-compatible body fields use
`with_raw_provider_json_for(&model, bytes)` or the `Value` convenience method and remain subject to
exact targeting, aggregate bounds, and the selected provider mode's protected-field policy.

Do not move credentials, endpoints, authorization headers, or transport policy into provider
options. The provider builder owns those settings.

## Language outcomes, stream failures, and terminal parity

`LanguageResponse` now represents successful provider generation only. Inspect its single
termination axis through `response.termination()`:

- `LanguageTermination::Completed(LanguageCompletionReason::...)` is a normal completion;
- `LanguageTermination::Incomplete(LanguageIncompleteReason::...)` is a successful but truncated
  or filtered generation.

Direct provider failure and cancellation return `LanguageCallError`, not a successful response with
a second status field. The error preserves the sanitized canonical `Error` and may expose a bounded
`partial()` containing observational text, reasoning, refusal, and usage. It never contains tool
execution authority, provider metadata, or replay material. Remove matches on the deleted
`LanguageResponseStatus` and `FinishReason` types.

An established language stream now settles exactly once through `StreamTerminal`. Provider errors
delivered over a successful HTTP SSE response appear as `StreamTerminal::Failed { error, .. }` with
the canonical `Error` contract. Applications should match `ErrorKind` and inspect bounded
`ResponseDiagnostics`; they should not parse provider messages or arbitrary error JSON. Exact
context-window and temporary-unavailability signals use the new
`ErrorKind::ContextWindowExceeded` and `ErrorKind::Unavailable` variants. Raw provider envelopes are
available only through the explicitly sensitive error accessor.

Failed and cancelled stream terminals expose `partial`, not a failed `LanguageResponse`. Usage
events now carry `UsageUpdate`; inspect `kind()` before reconciling a cumulative `Snapshot` or an
explicit `Delta`. Runtime performs that reconciliation per provider call, treats terminal usage as
the final snapshot, and charges budgets exactly once.

`SafeResponseHeaders`, `DiagnosticHeaderError`, `ResponseDiagnostics::headers`, and
`ResponseDiagnostics::with_headers` were removed. Header names cannot prove that provider-controlled
values are safe for default logs or serialization. Use the typed status, request ID, retry delay,
provider code/type/parameter, and truncation fields instead; raw headers remain available only from
the explicitly sensitive response accessor. `ResponseDiagnostics` now redacts request IDs from its
default `Debug` and serde serialization, which is deliberately not a lossless persistence format.
Read `request_id()` before serializing when the application needs that identifier, or retain the
explicitly sensitive response material inside its own trust boundary.

Chat Completions retains a trailing usage-only chunk before publishing its terminal response.
Responses streams compare an executable item shared by stable and terminal views using canonical
JSON semantics, and reject changes to its call ID, name, caller, or tool kind. Consumers no longer
need to normalize encoded tool-argument strings or reconcile disagreeing executable snapshots.

`OpenAiResponsesStreamFrame::native()` remains the exact provider event. Abbreviated terminal
events may require reconstruction from earlier completed items, so use
`canonical_terminal_response()` for the reconciled terminal `ResponseWire`. Accordingly,
`OpenAiResponsesStreamFrame::into_parts()` now returns
`(ResponsesStreamEvent, Vec<LanguageStreamEvent>, Option<ResponseWire>,
ResponsesReplayStatus)` instead of the former two-element tuple. The lower-level
`DecodedResponsesStreamFrame::into_parts()` similarly returns the native event, portable events,
and replay status. Replay status is pending before settlement and becomes available only when the
settled terminal resource has no replay-critical identity, reasoning-state, or provider-item
conflicts.

The branded `OpenAiProvider` always uses the official Responses wire baseline, including when a
caller supplies a custom endpoint. Use it only when the endpoint is OpenAI-faithful. Arbitrary
relays belong on `OpenAiCompatibleProvider`; compatible callers select a `ResponsesWireDialect` on
`OpenAiCompatibleProfile`. The generic default remains the strict OpenAI baseline, so abbreviated
terminal fields require an explicit, evidence-backed profile decision.

Advanced callers can preserve a previously validated transport policy, including an explicit RFC
6598 grant, without using a doc-hidden composition constructor:

```rust,ignore
use siumai_core::{ProviderId, ReplayDomain, ReplayDomainId};
use siumai_openai_compatible::{
    OpenAiCompatibleApiMode, OpenAiCompatibleProfile, ResponsesWireDialect,
};
use siumai_transport::EndpointConfig;

let endpoint = EndpointConfig::shared_address_space_explicit("http://100.64.0.10:8080/v1")?;
let profile = OpenAiCompatibleProfile::custom_endpoint(
    ProviderId::new("internal-relay")?,
    endpoint,
    ReplayDomain::custom(ReplayDomainId::new("internal-relay")?),
    OpenAiCompatibleApiMode::Responses,
)?
.with_responses_wire_dialect(ResponsesWireDialect::compatible());
```

`custom_endpoint` never turns a caller-controlled transport policy into a named-provider support
claim. Selecting `compatible()` is also not a generic “be permissive” switch: maintain the relay's
dialect fixture and keep the strict portable/executable reconciler.

The dialect may restore only its documented missing fields from one uniquely aligned completed
item. One shared reconciler still rejects changed portable text, refusal, role, executable
identity, tool name, caller ownership, or canonical JSON input. Provider-native bookkeeping drift
may keep the portable result while making native replay unavailable.

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

The official Anthropic builder keeps ownership of its replay audience and accepts only the
caller-owned scope when that is the missing boundary:

```rust,ignore
use siumai::{ReplayDomainId, providers::anthropic::AnthropicProvider};

let provider = AnthropicProvider::builder(credential)
    .with_caller_scope(ReplayDomainId::new("workspace-a")?)
    .build()?;
```

Prefer this helper for Anthropic Files-in-Messages and hosted-tool continuation on the official
endpoint. Do not reconstruct the provider-owned `anthropic-public-api` audience in application
code. Custom endpoints still require an explicit custom `ReplayDomain`; the caller-scope helper
only augments the selected audience.

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

## Explicit RFC 6598 relay endpoints

The transport no longer requires a loopback forwarder for a caller-authorized relay in RFC 6598
shared address space. Select the exact grant explicitly for each transport surface:

```rust,ignore
use siumai_transport::{EndpointConfig, ResourceUrl, WebSocketEndpoint};

let http = EndpointConfig::shared_address_space_explicit("http://100.64.0.10:8080/v1")?;
let websocket =
    WebSocketEndpoint::shared_address_space_explicit("ws://100.64.0.10:8080/v1/responses")?;
let resource = ResourceUrl::shared_address_space_explicit("http://100.64.0.10:8080/result")?;
```

This grant accepts only `100.64.0.0/10`, including equivalent IPv4-mapped IPv6 peers. It does not
authorize RFC 1918, loopback, link-local, adjacent public addresses, or mixed DNS answer sets.
Resource redirects remain on the exact original scheme, normalized host, and effective port. Using
cleartext HTTP or WS is an explicit caller trust decision; prefer TLS when the deployment supports
it.

## Node-scoped prompt caching

Prompt-cache intent no longer uses numeric message/content selectors or an untyped recursive map.
Attach a typed MiniMax cache marker to the semantic node it modifies:

```rust,no_run
use siumai::providers::minimax::MinimaxContentCache;
use siumai::{LanguageRequest, Message, MessagePart, MessageRole};

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

OpenAI prompt-cache markers use the same node-scoped principle. A marker represents one provider
wire breakpoint; it does not classify the node as historical or as a write candidate and does not
predict whether OpenAI will read or write cache state:

```rust,no_run
use siumai::providers::openai::prompt_cache::OpenAiContentOptions;
use siumai::{LanguageRequest, Message, MessagePart, MessageRole};

let cached = MessagePart::text("A stable reusable prefix")
    .with_provider_annotation(&OpenAiContentOptions::prompt_cache_breakpoint())?;
let request = LanguageRequest::new(vec![Message::new(
    MessageRole::User,
    [cached],
)]);
# Ok::<(), Box<dyn std::error::Error>>(())
```

Siumai enforces only structural rules it can prove locally: annotation ownership, duplicate
annotations on one node, wire shape, protected fields, and aggregate request bounds. Cache matching,
reads, writes, provider-side lookback windows, and implicit provider breakpoints remain provider
behavior. The old `OpenAiPromptCacheBreakpoint` coordinate type, historical/write-candidate roles,
and request-index helpers were removed because they predicted mutable provider state and became
invalid when middleware edited a request. See
[`ADR-0016`](../adr/0016-openai-prompt-cache-selection-remains-provider-owned.md).

`prompt_cache_options.ttl` and `prompt_cache_retention` remain distinct provider wire fields.
Current OpenAI guidance uses `ttl: "30m"` for GPT-5.6 and later and deprecates
`prompt_cache_retention` in favor of TTL. The deprecated field remains available for wire fidelity
to earlier documented models, but it is not presented as simultaneously applicable to GPT-5.6.
Siumai preserves explicit typed intent without turning dated model guidance into an execution
allowlist.

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
let responses = provider.responses_resource();
let voices = provider.voices();
# Ok::<(), Box<dyn std::error::Error>>(())
```

Each resource has typed request, response, identifier, status, and validation types under
`siumai::providers::minimax::resources`. Video and asynchronous speech expose explicit submit/query
operations; they do not start a hidden polling loop. Music and synchronous HTTP speech currently
return buffered non-streaming responses. MiniMax image and buffered speech now also have portable
`ImageModel` and `SpeechModel` adapters through `image(model)` and `speech_model(model_id)`; richer
provider controls remain on `images()` and `speech()`. `responses_resource()` counts input tokens
without starting generation, while `voices()` keeps clone/design/list/delete provider-owned.

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

The speculative `RegistryMiddleware` surface was removed. Delete calls to
`RegistryBuilder::middleware` and `RegistrySnapshot::middlewares`; Registry now performs only
deterministic lookup and its private canonical-route projection. `RegistryModelContext` remains
public so callers can inspect requested and canonical routes in typed resolution errors.

If an application needs call decoration, implement the relevant model-family trait on an ordinary
host-owned wrapper. Forward the inner model's complete `ModelDescriptor` and `route_id()` unchanged;
otherwise exact-target provider options, runtime route defaults, and route-aware diagnostics cannot
identify the configured route. Siumai does not provide or order a replacement decorator stack.

Custom providers now build non-empty registrations from a first family and add only disjoint
families:

```rust,ignore
let registration = ProviderRegistration::from_language(language_scope, language)
    .bind_transcription(transcription_scope, transcription)?;
```

The removed `ProviderRegistration::new(scope, policy).with_*` shape stored one provider-wide scope
and runtime model policy. Each replacement binding owns only an exact family scope and constructor.
Query it with an explicit family, for example `registration.api_mode(ModelFamily::Language)`, and
use `registration.for_family(ModelFamily::Rerank)` when a combined registration should be narrowed
before assigning a route. Same-family alternate API modes remain separate registrations.

`ModelPolicy`, `ProviderRegistration::evaluate`, and `Registry::evaluate` were removed. Registry
does not treat dated model catalogs, lifecycle hints, or capability claims as callability truth.
Hosts that need allowlists, lifecycle warnings, compliance, or availability decisions keep the
provider support manifest beside their route configuration and evaluate that evidence before
resolution. Unknown, private, and future model IDs remain constructible; stable request-shape
validation stays in the concrete provider and the remote API remains authoritative for mutable
product policy.

Facade `register_provider` is fallible at both seams. It reports normal Registry build errors and
`RegisterProviderError::NoPortableFamilyRegistration` when a provider is validly configured only for
provider-native resources or jobs. It never creates an empty registration.

Host-owned lifecycle policy is deliberately explicit. The following is schematic application code,
not a Siumai Registry API:

```rust,ignore
let support = provider_support_manifest(&provider);
let allowed = support.claims().any(|claim| {
    claim.family() == ModelFamily::Language && host_allowlist.contains(claim.model_id())
});
if !allowed {
    return Err(HostPolicyError::ModelNotAllowed);
}

let mut builder = Registry::builder();
builder.register_provider("primary", &provider)?;
let registry = builder.build()?;
let model = registry.language_model("primary:future-model")?;
```

The host may instead warn, route to a different configured instance, or skip the evidence check.
The important boundary is that Registry only resolves the host-selected registration and never
performs a generic advisory evaluation.

The standard MiniMax registration resolves the Messages mode. Register
`provider.chat_completions_registration()` or `provider.responses_registration()` explicitly under
separate host-owned route names when those modes are required. Each language-mode registration also
retains MiniMax's portable image and speech families; use `for_family(ModelFamily::Language)` when a
route should expose language only.

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
- xAI keeps Responses as its primary language mode and exposes Chat Completions explicitly. It now
  implements portable image generation, buffered speech, and final transcription, while Files and
  asynchronous video generation live in provider-owned typed resources. Realtime and Batch remain
  deliberately deferred instead of being inferred from OpenAI compatibility.
- Groq now combines verified Chat Completions/Responses with final transcription, buffered Orpheus
  speech, typed Remote MCP options and outputs, and provider-owned URL-audio transcription and
  translation. Files and Batch remain unclaimed until their official contracts are consistent
  enough for a bounded implementation.
- Cohere retains embeddings and reranking; the old chat surface is removed. The new model-less
  `provider.transcriptions()` resource exposes the official v2 audio transcription endpoint
  without inventing a model identifier or Registry family.
- Cohere model hints now track Embed v4 and Rerank v4/v3. Obsolete Embed v2 hints are removed;
  textual future or legacy IDs remain callable through the open model-ID contract.
- Gemini keeps portable text embedding on stable v1 and adds the separate provider-native
  `provider.multimodal_embedding(model)` v1beta path for ordered text/image/audio/video/PDF input.
  Existing portable embedding requests are not widened into a cross-provider multimodal type.
- Deepgram publishes current Nova-3/Nova-2 prerecorded hints and a portable buffered Aura speech
  family. Legacy Base, Enhanced, and Nova constants remain removed, and every model ID stays open.
- ElevenLabs retains buffered speech synthesis and restores final-result/batch transcription over
  the provider-owned Speech-to-Text wire contract. Realtime transcription remains intentionally
  outside the portable family surface.
- OpenAI retains configured Chat Completions, Responses, Responses resources, opt-in Realtime, and
  the separate experimental Responses WebSocket session feature. It also exposes portable
  embedding, image-generation, buffered speech, and final-result transcription handles, plus typed
  Conversations, Files, Vector Stores, and Skills resources.
  Moderation and broad legacy compatibility resources remain intentionally outside this release
  slice.
- The generic OpenAI-compatible engine retains explicit generic and custom endpoints only. Kimi is
  owned by `siumai-provider-moonshotai`; Volcengine ARK is owned by
  `siumai-provider-volcengine`. Their typed options, model advisories, and support evidence no
  longer appear below the compatibility-engine namespace. Unverified GLM, Qianfan, Hunyuan,
  SiliconFlow, DeepInfra, and other named presets are not shipped as support claims.
- Moonshot AI now exposes Kimi Partial Mode through a final-assistant message annotation and a
  typed Files lifecycle (`files()`). Volcengine ARK now exposes `images()` and `video_tasks()` as
  provider-owned native surfaces, while `image(model)` implements the portable `ImageModel`
  contract. No generic Video job trait was reintroduced.

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

Runtime durable snapshots now use schema version 8 and durable execution ABI
`siumai-runtime-durable-v7`. Version 7 and earlier snapshots are rejected at the version envelope
before typed payload decoding; Siumai does not migrate them automatically. Drain beta-era workers
before the upgrade and recreate required runs from trusted application history rather than
synthesizing journal events, provider provenance, deferred resolution, or terminal state.

Schema v8 records the validated tool journal and the exact provider-deferred ledger, and removes the
unused tool `dispatch_id`. Provider-deferred identity is the complete `ProviderScope` plus the
bounded correlation identifier. Observations update in first-seen order, resolution is monotonic,
and failed, cancelled, or unexpectedly closed streams cannot commit staged observations as
resumable work. `SuspensionReason::AwaitingProvider` now exposes only a pending count instead of raw
state identifiers.

Snapshot internals are now inspection-only public APIs. Replace direct enum construction and
pattern matching with `ResumePoint::kind`, `ResumePoint` accessors, `SnapshotTerminal::kind`,
terminal accessors, and the read-only journal/deferred views. Constructors and mutators for
snapshots, checkpoints, pending states, execution events, execution logs, provider-state
projections, and runtime-owned fingerprints are no longer public. Real external store adapters keep
the public `RunStore`, lease, revision, and `StoredRun` construction seams.

Every initial, ordinary, approval, provider-suspension, recovery, and terminal checkpoint now passes
through one runtime-owned writer. The writer validates the successor and enforces
`RunBudget::max_snapshot_bytes` before store CAS. External stores must also bound raw serialized
input before deserialization and provide confidentiality, integrity and authenticity, tenant/run
isolation, access control, and rollback or revision protection. Serialized snapshots and explicit
provider-state payload accessors retain sensitive replay bytes; they are not safe diagnostics.

## Gateway projections

Gateway JSON is intentionally a bounded, sanitized projection rather than a provider/runtime
inspection surface. The breaking `model.projection.loss.data` payload for a deferred provider
correlation has changed from a raw identifier:

```json
{ "id": "deferred-correlation-id" }
```

to an aggregate marker:

```json
{ "provider_deferred": { "pending": 1 } }
```

Likewise, terminal run-report projections may expose provider-deferred state only as
`{ "total": 2, "resolved": 1, "pending": 1 }` counts. Correlation identifiers and provider
payloads are intentionally unavailable through ordinary gateway JSON. Access them through an
explicit provider- or runtime-sensitive API before projection when the application has an
appropriate trust boundary.

Tool loops preserve caller-supplied model-visible tools, append trusted
local bindings in deterministic name order, and bind snapshots and approvals to the complete
annotated visible catalog plus exact local execution identities. Snapshots carrying an older
execution ABI or tool catalog fingerprint are rejected explicitly; restart those beta-era runs
from trusted application history. Caller-visible tools remain non-executable unless they are
defined by the host as `ToolBinding`s instead. Provider-hosted tools remain provider-owned, and a
caller-visible definition that duplicates a local binding name fails before model I/O.

## Anthropic Message Batches and Skills

`AnthropicMessageBatches::results` now returns `AnthropicBatchResultsStream` instead of a buffered
`Bytes` body. The stream incrementally decodes unordered JSONL records, applies encoded and decoded
resource budgets while constructing each JSON value, and reports one typed terminal error for a
malformed or truncated record. Use `futures_util::StreamExt` and correlate records through
`AnthropicBatchResult::custom_id`:

```rust,ignore
use futures_util::StreamExt;

let mut results = provider.message_batches().results(batch_id).await?;
while let Some(result) = results.next().await {
    let result = result?;
    println!("{}: {}", result.custom_id, result.status().as_str());
}
```

Batch and Skills response discriminants now use bounded open wrappers such as
`AnthropicBatchProcessingStatus`, `AnthropicBatchResultStatus`, `AnthropicSkillResponseType`, and
`AnthropicSkillSource`. Replace `Option<String>::as_deref()` calls with the wrapper's `as_str()`
accessor. Unknown future values remain available; they are not converted into a closed allowlist.

`AnthropicBatchDeleteResult` now follows the provider response shape directly: `object_type` is a
required bounded value and `deleted` is replaced by `is_deleted()`. Old persisted delete payloads
without a `type` field are no longer accepted by serde.

Prompt caching and server-side fallbacks remain available in Message Batches. Current Anthropic
guidance excludes speed/Fast mode, so batch construction rejects both top-level speed and a fallback
that requests speed. The results stream does not issue a hidden retrieve request to infer the
expected result count. Callers that require result-count reconciliation should retrieve the batch
explicitly and compare its request counts with the streamed records.

The Anthropic Skills support claim is `claimed slice complete` for bounded create uploads and the
implemented metadata/version operations; it is not a `provider platform complete` claim. Skills
expose list, retrieve, delete, version-list, version-create, version-retrieve, and version-delete
operations. `versions` remains the first-page convenience. Use `versions_page` with
`AnthropicSkillVersionListQuery` to follow `next_page`. Multipart file uploads require one common
top-level directory and a root `SKILL.md`; ZIP input remains opaque and server-validated. Siumai
does not inspect ZIP archive contents locally. Skill version-content download is `intentionally
deferred`. `AnthropicSkillList::has_more` and `AnthropicSkillVersionList::has_more` are now
methods: replace field reads with `has_more()`, whose value is derived from `next_page.is_some()`.
The former `AnthropicSkills::upload` and `upload_with_options` aliases were removed; use `create`
and `create_with_options` respectively.

## Migration checklist

- Replace every `siumai::families::*` import with the matching root `language`, `embedding`,
  `rerank`, `image`, `speech`, or `transcription` module.
- Replace the seven facade `*_with_options` helpers with
  `family::call(...).with_options(options)?.<terminal>().await`; attach typed provider intent with
  `with_provider_options` on the same bound call.
- Import runtime single-call helpers explicitly from `siumai::runtime::{generate, stream}` rather
  than the facade root or prelude.
- Replace `siumai_runtime::AgentInput` and downstream conversions into it with
  `siumai_core::LanguageInput` or the facade re-export `siumai::LanguageInput`.
- Treat `LanguageResponse::output_text()` as an optional display projection only; retain the
  complete response for termination, usage, warnings, metadata, non-text content, and history
  projection.
- Pass concrete and Registry-resolved models through the same root family functions, and retain the
  concrete provider separately for native resources instead of attempting a Registry downcast.
- Replace `MinimaxConfig` and `MinimaxClient` with `MinimaxCredential` and `MinimaxProvider`.
- Construct a family model explicitly and keep the configured provider long-lived.
- Replace compatibility chat request types with `LanguageRequest`, `Message`, and `MessagePart`.
- Replace direct `ToolCall` field construction with `ToolCall::local` and checked accessors.
- Use role-safe message constructors and project responses with
  `LanguageResponse::project_assistant_history()` before appending assistant history.
- Move provider call controls into the matching typed MiniMax options and bind them through the
  root family call builder; use `CallOptions::with_provider_options_for` only for lower-level option
  assembly.
- Replace public provider-option origin/layer/merger code with ordered exact-target patches; keep
  route, model, step, and call precedence inside runtime assembly.
- Replace `LanguageResponseStatus`/`FinishReason` matches with `LanguageTermination`; handle direct
  failures through `LanguageCallError`, stream failure/cancellation through bounded `partial`, and
  usage events through `UsageUpdate`.
- Move prompt-cache intent onto typed message, content, or tool annotations.
- Update native Responses stream destructuring for the canonical terminal response returned by
  `OpenAiResponsesStreamFrame::into_parts()`.
- Move custom Responses terminal contractions to
  `OpenAiCompatibleProfile::with_responses_wire_dialect`; the branded OpenAI builder no longer
  accepts a wire-dialect override.
- Use `OpenAiCompatibleProfile::custom_endpoint` when a compatible relay needs a prevalidated
  custom transport policy such as an RFC 6598 grant; do not model an arbitrary relay as branded
  OpenAI.
- Enable `openai-responses-websocket` only when the application needs persistent provider-owned
  Responses turns; configure a WebSocket endpoint explicitly for custom HTTP providers.
- Acquire files, image, video, music, and speech APIs from `MinimaxProvider`.
- Add Registry only for explicit local routing, and register each non-default API mode separately.
- Replace provider-level HTTP limits/retry/timeout/observer setters with one
  `ProviderHttpTransportSettings` value and `with_http_transport_settings(...)`.
- Resolve caller timing with `CallOptions::with_timeout` and narrow retries with
  `CallOptions::with_max_attempts`; do not treat either as replay permission.
- Keep Realtime, Responses WebSocket, external-download, media-job, and MCP controls independent
  from stateless provider HTTP settings.
- Move model lifecycle, allowlist, availability, compliance, and fallback decisions out of removed
  `ModelPolicy`/`Registry::evaluate` execution paths and into explicit host policy.
- Replace provider-wide `scope()` or `platform()` queries with `provider_id()` or an exact model or
  registration family scope.
- Keep account, region, availability, pricing, and fallback policy in the host application.
- Declare an explicit custom replay domain for every custom language endpoint, and separate material
  accounts, projects, workspaces, or deployments with non-secret caller scopes.
- Recreate pre-version-8 runtime snapshots from trusted application history.
- Replace removed provider/features with an explicitly supported slice or a generic compatible
  endpoint only when protocol compatibility is sufficient for the application.

See [`../providers/minimax.md`](../providers/minimax.md) for dated support evidence and deliberate
limitations.
