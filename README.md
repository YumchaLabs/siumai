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

Enable only the provider and integration features that the application uses:

```toml
[dependencies]
siumai = { version = "0.11.0-beta.9", default-features = false, features = ["minimax"] }
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
| `openai` | Chat Completions, Responses, and Responses resources |
| `openai-realtime` | Experimental OpenAI Realtime support |
| `anthropic` | Messages and Anthropic-native resources |
| `google` | Experimental Gemini Interactions image generation |
| `google-vertex-anthropic` | Anthropic Messages on Google Vertex AI |
| `alibaba` | Chat, Responses, embeddings, and experimental Wan video |
| `moonshotai` | Moonshot AI's Kimi Chat Completions product surface |
| `volcengine` | Volcengine ARK Chat Completions and Responses modes |
| `openai-compatible` | Explicit generic or custom OpenAI-compatible endpoints |
| `groq` | Chat, Responses, and transcription |
| `xai` | Responses and Chat Completions language modes |
| `minimax` | Three language modes plus files, image, video, music, and speech |
| `deepseek` | Chat and Responses language modes |
| `cohere` | Embeddings and reranking |
| `deepgram` | Final-result transcription |
| `elevenlabs` | Speech synthesis |

See the [provider support policy](docs/providers/support-policy.md) for fidelity, stability, and
host-control-plane boundaries. Model identifiers remain open; constants are dated hints rather
than allowlists.

## Choose the narrowest public surface

- Start with a configured provider when protocol modes, provider options, or native resources
  matter.
- Pass its model handles through `siumai::families::*` when an operation is portable.
- Add Registry only when the host needs deterministic local route lookup.
- Add runtime only when the host needs provider-neutral multi-step orchestration.

Provider-specific behavior stays available through typed call options, typed annotations attached
to semantic nodes, or provider-native resources. Only semantics demonstrated to be portable belong
in shared family requests.

## MiniMax provider-direct example

MiniMax uses Anthropic-compatible Messages as its recommended language mode:

```rust,no_run
use siumai::families::language;
use siumai::providers::minimax::{
    MinimaxCredential, MinimaxMessagesOptions, MinimaxProvider, MinimaxServiceTier,
    MinimaxThinking, models,
};
use siumai::{CallOptions, ContentPart, LanguageRequest, Message, MessageRole};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let provider = MinimaxProvider::builder(MinimaxCredential::api_key(
        std::env::var("MINIMAX_API_KEY")?,
    ))
    .build()?;
    let model = provider.language(models::MINIMAX_M3)?;

    let provider_options = MinimaxMessagesOptions::new()
        .with_thinking(MinimaxThinking::Adaptive)
        .with_service_tier(MinimaxServiceTier::Standard)
        .provider_options()?;
    let response = language::generate_with_options(
        &model,
        LanguageRequest::new(vec![Message::text(
            MessageRole::User,
            "Hello MiniMax!",
        )]),
        CallOptions::default().with_provider_options(provider_options),
    )
    .await?;

    for part in response.content() {
        if let ContentPart::Text { text } = part {
            print!("{text}");
        }
    }
    Ok(())
}
```

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
retains its own exact protocol scope and request policy.

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
explicit provider-owned choices. `register_provider` returns a typed error when a valid provider
configuration exposes only provider-native resources or jobs. Combined registrations can be narrowed
with `for_family` before assigning a route.

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
- [Registry contract](docs/architecture/registry.md)
- [Transport contract](docs/architecture/transport-contract.md)
- [Provider support policy](docs/providers/support-policy.md)
- [Google image support evidence](docs/providers/google.md)
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
