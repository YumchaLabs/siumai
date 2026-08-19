# `siumai` facade

`siumai` is the curated public facade for the Siumai workspace. It re-exports six small
provider-neutral model-family contracts and opt-in provider-owned APIs without flattening native
resources or protocol controls into a least-common-denominator client.

The facade has no provider enabled by default. Select only the providers and orchestration layers
the application uses:

```toml
[dependencies]
siumai = { version = "0.11.0-beta.10", default-features = false, features = ["openai"] }
```

## Canonical family calls

The six stable provider-neutral entry points live at the crate root: `language`, `embedding`,
`rerank`, `image`, `speech`, and `transcription`. Each module has one concise default operation and
one `call` builder for explicit `CallOptions` or typed provider options. The model handle may be a
concrete provider model or a Registry-resolved trait object; application call code does not need to
match on the provider.

Language accepts a string, one `Message`, a message list, or a complete `LanguageRequest` through
the same path:

```rust,no_run
use siumai::{LanguageCallError, LanguageModel, LanguageResponse, language};

async fn answer<M>(model: &M) -> Result<LanguageResponse, LanguageCallError>
where
    M: LanguageModel + ?Sized,
{
    let response = language::generate(model, "Explain bounded streaming in one sentence.").await?;
    if let Some(text) = response.output_text() {
        println!("{text}");
    }
    Ok(response)
}
```

The return value is still the complete `LanguageResponse`. `output_text()` is an optional,
display-oriented concatenation of canonical text parts; it excludes reasoning, refusals, tools,
citations, media, and provider-native state. Inspect `content()`, `termination()`, `usage()`,
`warnings()`, and provider metadata when those semantics matter, and use
`project_assistant_history()` rather than text projection when constructing a continuation.

Provider-specific intent binds to the same live model through the ordinary call builder, while
native resources remain on the concrete provider:

```rust,no_run
# #[cfg(feature = "openai")]
# async fn call_openai() -> Result<(), Box<dyn std::error::Error>> {
use siumai::language;
use siumai::providers::openai::models::GPT_5_6;
use siumai::providers::openai::responses::OpenAiResponsesOptions;
use siumai::providers::openai::{OpenAiCredential, OpenAiProvider};

let provider = OpenAiProvider::builder(OpenAiCredential::api_key("example-key")).build()?;
let model = provider.responses(GPT_5_6)?;
let provider_options = OpenAiResponsesOptions {
    instructions: Some("Keep this provider intent exact.".to_string()),
    ..OpenAiResponsesOptions::default()
};
let response = language::call(&model, "Summarize the facade boundary.")
    .with_provider_options(&provider_options)?
    .generate()
    .await?;

let _complete_content = response.content();
let _files = provider.files();
# Ok(())
# }
```

Typed annotations follow the same ownership rule but attach to the message, content part, or tool
they modify. Registry deliberately does not downcast a resolved family model back into a concrete
provider; retain the configured provider separately when the application needs files, batches,
sessions, or other native APIs.

Every provider feature activates the narrow `siumai::transport` configuration namespace. Enable
`transport` by itself when an assembly crate only needs to construct shared provider HTTP settings.
The namespace intentionally excludes authenticated execution, request plans, raw responses, and
socket primitives.

```rust,no_run
# #[cfg(feature = "openai")]
# fn build_openai_provider() -> Result<(), Box<dyn std::error::Error>> {
use std::time::Duration;
use siumai::providers::openai::{OpenAiCredential, OpenAiProvider};
use siumai::transport::{ProviderHttpTransportSettings, RetryPolicy};

let http_settings = ProviderHttpTransportSettings::default()
    .with_retry_policy(RetryPolicy::new(2)?)
    .with_call_timeout(Duration::from_secs(120))?;
let _provider = OpenAiProvider::builder(OpenAiCredential::api_key("example-key"))
    .with_http_transport_settings(http_settings)
    .build()?;
# Ok::<(), Box<dyn std::error::Error>>(())
# }
```

Select a trusted forward proxy explicitly; constructing the route performs no environment lookup
or network I/O:

```rust
use siumai::transport::{HttpTransportRoute, ProviderHttpTransportSettings, ProxyEndpoint};

let route = HttpTransportRoute::trusted_connect(
    ProxyEndpoint::https("https://proxy.example.com")?,
);
let settings = ProviderHttpTransportSettings::default().with_route(route)?;

assert!(settings.route().proxy().is_some());
# Ok::<(), Box<dyn std::error::Error>>(())
```

Transport observers receive only bounded structural attempt events. They cannot inspect URLs,
headers, credentials, bodies, prompts, outputs, or provider identity.

Direct remains the default and ignores environment/system proxy configuration. A custom endpoint
is a reverse gateway, not a forward proxy. The alternative is one explicit trusted CONNECT route
for public HTTPS provider origins. Proxy Basic authentication and provider authentication remain
separate, redacted audiences; rotate an immutable proxy credential by rebuilding the provider.
WebSocket, Realtime, and provider-returned external-download routes remain Direct-only.

Complete portable requests use the same core types regardless of provider:

```rust
use siumai::{LanguageRequest, Message, MessageRole};

let request = LanguageRequest::new(vec![Message::text(
    MessageRole::User,
    "Explain bounded streaming in one sentence.",
)]);

assert_eq!(request.messages.len(), 1);
```

Provider-specific behavior remains available under `siumai::providers`, including typed options,
annotations, native resources, and experimental session APIs behind their owning feature flags.
Enable `registry` for deterministic lookup over caller-configured providers and `runtime` for
provider-neutral multi-step execution.

The packaged facade includes four compile-checked examples:

- `examples/openai_flagship.rs` combines an exact-target Responses option, the portable language
  family, and the provider-owned Conversations lifecycle;
- `examples/anthropic_flagship.rs` combines Messages options, scope-bound Files-in-Messages,
  assistant-history replay, and a provider-owned Skills metadata list;
- `examples/provider_switching.rs` passes concrete, erased, and Registry-resolved models through
  one application function while preserving typed options, annotations, complete response
  metadata, and concrete provider resources;
- `examples/trusted_connect_route.rs` constructs a trusted route through curated facade APIs and
  proves how a provider builder consumes the resulting settings without credentials or I/O.

The flagship examples are offline by default and perform network calls only after their provider
credential environment variable is set. The provider-switching and trusted-route examples perform
no network I/O.

See the [repository README](https://github.com/YumchaLabs/siumai#readme),
[migration guide](https://github.com/YumchaLabs/siumai/blob/main/docs/migration/siumai-next.md), and
[provider support policy](https://github.com/YumchaLabs/siumai/blob/main/docs/providers/support-policy.md)
for the primary user journey and current support evidence.
