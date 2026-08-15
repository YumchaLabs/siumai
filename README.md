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
siumai = { version = "0.11.0-beta.10", default-features = false, features = ["minimax"] }
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

## Flagship OpenAI and Anthropic journeys

The facade ships two compile-checked, offline-by-default examples:

- [`openai_flagship.rs`](siumai/examples/openai_flagship.rs) combines an exact-target typed
  Responses option, the portable language family, and a provider-owned Conversations read;
- [`anthropic_flagship.rs`](siumai/examples/anthropic_flagship.rs) combines current Messages
  options, scope-bound Files-in-Messages, canonical assistant-history replay, and a provider-owned
  Skills metadata list.

Compile them independently with only their documented provider feature:

```text
cargo check -p siumai --example openai_flagship --no-default-features --features openai -j 1
cargo check -p siumai --example anthropic_flagship --no-default-features --features anthropic -j 1
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
- Pass its model handles through `siumai::families::*` when an operation is portable.
- Add Registry only when the host needs deterministic local route lookup.
- Add runtime only when the host needs provider-neutral multi-step orchestration.

Provider-specific behavior stays available through typed call options, typed annotations attached
to semantic nodes, or provider-native resources. Only semantics demonstrated to be portable belong
in shared family requests. Call-level provider options are bounded ordered patches and normally bind
to the exact configured model instance; host route/model/step/call precedence remains private to
runtime instead of becoming part of every provider API.

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
        .with_service_tier(MinimaxServiceTier::Standard);
    let options = CallOptions::default().with_provider_options_for(&model, &provider_options)?;
    let response = language::generate_with_options(
        &model,
        LanguageRequest::new(vec![Message::text(
            MessageRole::User,
            "Hello MiniMax!",
        )]),
        options,
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
