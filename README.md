# Siumai — Provider-faithful AI model interfaces for Rust

[![Crates.io](https://img.shields.io/crates/v/siumai.svg)](https://crates.io/crates/siumai)
[![Documentation](https://docs.rs/siumai/badge.svg)](https://docs.rs/siumai)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](https://github.com/YumchaLabs/siumai/blob/main/LICENSE)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/YumchaLabs/siumai)

Siumai (烧卖) is a type-safe Rust workspace for integrating AI model providers. It offers
provider-faithful typed access to explicitly retained provider scopes and small provider-neutral
model-family interfaces for portable application code. The unified family API and Registry are
optional convenience layers, not replacements for provider-specific capabilities.

## What Siumai provides

- Provider-direct construction, typed options, annotations, metadata, and native resources
- Provider-neutral family traits for language, embeddings, images, reranking, speech, and
  transcription
- Established language streams with explicit terminal outcomes and cancellation
- An optional immutable Registry for routing caller-configured providers
- An optional runtime for tool loops, structured output, approvals, budgets, and durable runs
- Shared transport and protocol crates with bounded, sanitized error handling

## Install

The examples in this README document the unreleased facade on the repository's `main` branch.
Until the next crates.io beta containing this API is published, use the Git dependency below.
Users staying on `0.11.0-beta.10` should follow the old side of the
[migration map](docs/migration/siumai-next.md#facade-convergence-after-0110-beta10).

Enable only the provider and integration features that the application uses:

```toml
[dependencies]
siumai = { git = "https://github.com/YumchaLabs/siumai.git", default-features = false, features = ["minimax"] }
tokio = { version = "1", features = ["rt-multi-thread", "macros"] }
```

The facade's default features are `registry` and `runtime`; no provider is enabled by default. Add
`registry` or `runtime` explicitly when using `default-features = false`.

The current workspace is a breaking public API reset. Existing users should read
[`docs/migration/siumai-next.md`](docs/migration/siumai-next.md) before updating.

## Provider features

Provider features describe the exact retained slice, not every product sold under a vendor name:

| Feature | Current public scope |
|---|---|
| `openai` | Responses and Chat language, embeddings, image generation, buffered speech, final-result transcription, and typed Conversations/Files/Vector Stores/Skills resources |
| `openai-realtime` | Experimental provider-owned OpenAI Realtime bootstrap and session transport |
| `openai-responses-websocket` | Experimental persistent OpenAI Responses WebSocket sessions; also enables `openai` |
| `anthropic` | Messages plus Files, Message Batches, token counting, and Skills metadata/version CRUD with bounded uploads |
| `google` | Gemini Interactions/GenerateContent language, text and native multimodal embedding, image, speech, Files, and Veo |
| `google-vertex-anthropic` | Anthropic Messages on Google Vertex AI |
| `alibaba` | Chat, Responses, Anthropic-compatible Messages, embeddings, and experimental Wan video |
| `moonshotai` | Moonshot AI's Kimi Chat Completions, Partial Mode, and typed Files lifecycle |
| `volcengine` | Volcengine ARK Chat Completions/Responses, portable Image, Remote MCP, and typed Video tasks |
| `openai-compatible` | Explicit generic or custom OpenAI-compatible endpoints |
| `groq` | Chat, Responses, transcription, buffered Orpheus speech, Remote MCP, and URL-audio/translation resources |
| `xai` | Responses-primary language, explicit Chat Completions, image, speech, transcription, Files, and video jobs |
| `minimax` | Three language modes, portable image/speech, Responses input-token counting, and typed files/media/voice resources |
| `deepseek` | Chat, beta Chat strict/prefix, Responses, and Anthropic-compatible Messages language modes |
| `cohere` | Embeddings, reranking, and provider-native audio transcription |
| `deepgram` | Final-result transcription and buffered Aura speech synthesis |
| `elevenlabs` | Buffered speech synthesis and final-result/batch transcription |

See the [provider support policy](docs/providers/support-policy.md) for fidelity, stability, and
host-control-plane boundaries. Model identifiers remain open; constants are dated hints rather
than allowlists.

Each row is a `claimed slice complete` inventory for this release, not a `provider platform
complete` claim. Surfaces outside a row's exact scope are `intentionally deferred` and remain
available for future provider-owned additions without widening the portable core.

## Provider HTTP transport settings and trusted CONNECT

Provider features automatically enable the facade's narrow `siumai::transport` namespace. The
standalone `transport` feature exposes only provider HTTP configuration and payload-free attempt
observation; authenticated execution, request plans, raw responses, resource downloaders, and
socket primitives remain in their owning crates.

```rust
use std::time::Duration;
use siumai::transport::{ProviderHttpTransportSettings, RetryPolicy};

let settings = ProviderHttpTransportSettings::default()
    .with_retry_policy(RetryPolicy::new(2).unwrap())
    .with_call_timeout(Duration::from_secs(120))
    .unwrap();

assert_eq!(settings.retry_policy().max_attempts(), 2);
```

The default route is Direct. Select a trusted forward proxy explicitly without consulting process
proxy variables:

```rust
use siumai::transport::{HttpTransportRoute, ProviderHttpTransportSettings, ProxyEndpoint};

let route = HttpTransportRoute::trusted_connect(
    ProxyEndpoint::https("https://proxy.example.com").unwrap(),
);
let settings = ProviderHttpTransportSettings::default()
    .with_route(route)
    .unwrap();

assert!(settings.route().proxy().is_some());
```

The same cloneable settings value can be reused across configured providers. Its observer events
contain structural attempt and retry state only—never URLs, headers, credentials, payloads, or
provider identity.

Direct validates provider DNS and peers locally. Trusted CONNECT validates the proxy endpoint and
peer, then trusts that proxy for destination DNS/peer selection while preserving the logical
provider URL, inner TLS, credential audience, replay proof, deadlines, and bounds. Proxy Basic
authentication is separate from provider authentication and rotates by rebuilding the configured
provider. Streamable HTTP MCP reuses only the route type through `McpClientConfig`; WebSocket,
Realtime, and provider-returned external downloads remain Direct-only.

## Verified facade journeys

The facade ships four compile-checked examples:

- [`openai_flagship.rs`](siumai/examples/openai_flagship.rs) combines an exact-target typed
  Responses option, the portable language family, and a provider-owned Conversations read;
- [`anthropic_flagship.rs`](siumai/examples/anthropic_flagship.rs) combines current Messages
  options, scope-bound Files-in-Messages, canonical assistant-history replay, and a provider-owned
  Skills metadata list;
- [`provider_switching.rs`](siumai/examples/provider_switching.rs) passes concrete, erased, and
  Registry-resolved language models through one application function while retaining typed OpenAI
  and Anthropic options, annotations, metadata views, and concrete provider resources;
- [`trusted_connect_route.rs`](siumai/examples/trusted_connect_route.rs) constructs the trusted
  route and compiles a provider settings handoff without credentials or network I/O.

Compile or execute them independently with only their documented feature set:

```text
cargo check -p siumai --example openai_flagship --no-default-features --features openai -j 1
cargo check -p siumai --example anthropic_flagship --no-default-features --features anthropic -j 1
cargo run -p siumai --example provider_switching --no-default-features --features openai,anthropic,registry -j 1
cargo check -p siumai --example trusted_connect_route --no-default-features --features openai -j 1
```

OpenAI Responses WebSocket is intentionally provider-owned rather than a portable family. Enable
`openai-responses-websocket`, acquire a Responses model, call `model.websocket()?`, and connect the
returned configuration. One connection accepts one generated turn at a time, supports sequential
continuation and native-only warm-up, and uses the same canonical Responses decoder as HTTP SSE.
Custom HTTP providers must configure a WebSocket endpoint explicitly and do not inherit OpenAI's
official session support claim.

## Choose the narrowest public surface

- Start with a configured provider when protocol modes, provider options, or native resources
  matter.
- Pass its model handles through the root `siumai::{language, embedding, rerank, image, speech,
  transcription}` modules when an operation is portable.
- Add Registry only when the host needs deterministic local route lookup.
- Add runtime only when the host needs provider-neutral multi-step orchestration.

Provider-specific behavior stays available through typed call options, typed annotations attached
to semantic nodes, or provider-native resources. Only semantics demonstrated to be portable belong
in shared family requests. Call-level provider options are bounded ordered patches and normally bind
to the exact configured model instance; host route/model/step/call precedence remains private to
runtime instead of becoming part of every provider API.

Registry resolution returns the same family trait object accepted by the root modules. Resolve a
route once and pass that live handle to ordinary application code; Registry does not recover a
concrete provider through downcasting. Keep the configured concrete provider beside Registry when
the application also needs native files, batches, sessions, or media jobs.

## Canonical language call with MiniMax

MiniMax uses Anthropic-compatible Messages as its recommended language mode:

```rust,no_run
use siumai::language;
use siumai::providers::minimax::{
    MinimaxCredential, MinimaxMessagesOptions, MinimaxProvider, MinimaxServiceTier,
    MinimaxThinking, models,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let provider = MinimaxProvider::builder(MinimaxCredential::api_key(
        std::env::var("MINIMAX_API_KEY")?,
    ))
    .build()?;
    let model = provider.language(models::MINIMAX_M3)?;

    let provider_options = MinimaxMessagesOptions::new()
        .with_thinking(MinimaxThinking::Adaptive)
        .with_service_tier(MinimaxServiceTier::Standard);
    let response = language::call(&model, "Hello MiniMax!")
        .with_provider_options(&provider_options)?
        .generate()
        .await?;

    if let Some(text) = response.output_text() {
        print!("{text}");
    }
    Ok(())
}
```

The call returns the complete `LanguageResponse`: content, termination, usage, warnings, and
provider metadata remain available. `output_text()` is only a display-oriented concatenation of
canonical text parts. It excludes reasoning, refusals, tools, citations, media, and
provider-native state, and it does not replace `project_assistant_history()` when building the next
request.

`provider.language(model)` and `provider.messages(model)` both select Messages. Use
`provider.chat_completions(model)` or `provider.responses(model)` only when that wire API is an
explicit requirement. Each mode has a matching typed options type and rejects options from another
mode.

Messages prompt-cache breakpoints use typed `MinimaxMessageCache`, `MinimaxContentCache`, or
`MinimaxToolCache` annotations on the node they modify. The same configured provider owns native
resources through:

- `provider.files()`;
- `provider.images()`;
- `provider.video()`;
- `provider.music()`;
- `provider.speech()`.

These resources are not flattened into a universal client because their request shapes, result
types, and task lifecycles are provider-specific.

See [`docs/providers/minimax.md`](docs/providers/minimax.md) for dated support evidence, API-mode
stability, resource boundaries, and deliberate streaming limitations.

## Optional Registry

Registry stores immutable registrations from providers the caller has already configured. It does
not discover remote models, read hidden credentials, choose a region, or apply business fallback
policy. One route may expose several disjoint model families from the same provider; each family
retains its own exact protocol scope and configured factory.

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

The default MiniMax registration uses Messages. Chat Completions and Responses registrations remain
explicit provider-owned choices while retaining the same portable image and speech families.
`register_provider` returns a typed error when a valid provider configuration exposes only
provider-native resources or jobs. Combined registrations can be narrowed with `for_family` before
assigning a route.

## Provider plane and host control plane

Siumai owns request correctness, authentication, technical endpoints, wire behavior, response
decoding, and provider-native resource operations. The host application owns account and region
selection, aliases, default models, current commercial availability, pricing, quota, compliance,
health, weights, and fallback.

A provider may accept a technical region, project, deployment, or endpoint when the remote API needs
it for addressing or signing. That input is not an SDK-maintained availability catalog.

## Documentation

- [Repository architecture](docs/architecture/overview.md)
- [Public API and extension policy](docs/architecture/public-api.md)
- [Facade family call ownership decision](docs/adr/0019-facade-family-call-ownership.md)
- [Registry contract](docs/architecture/registry.md)
- [Transport contract](docs/architecture/transport-contract.md)
- [Provider support policy](docs/providers/support-policy.md)
- [Google Gemini support evidence](docs/providers/google.md)
- [Contributing](CONTRIBUTING.md)
- [Documentation index](docs/README.md)

## Development

Run Cargo commands serially and prefer focused package checks:

```bash
cargo fmt --all -- --check
cargo nextest run -p siumai-provider-minimax --all-features --test-threads 1
cargo clippy -p siumai-provider-minimax --all-targets --all-features -j 1 -- -D warnings
```

Default tests are deterministic, offline, and secret-free. Do not run live, credentialed, billable,
or destructive provider tests without explicit authorization.

## Acknowledgements

Siumai draws product inspiration from the [Vercel AI SDK](https://github.com/vercel/ai) while keeping
its public ownership, traits, builders, errors, feature flags, and async boundaries Rust-first.

## Changelog and license

See [`CHANGELOG.md`](CHANGELOG.md) for release history. Siumai is licensed under either the Apache
License, Version 2.0, or the MIT license, at your option.
