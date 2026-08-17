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

Portable requests use the same core types regardless of provider:

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

The packaged facade includes three compile-checked examples:

- `examples/openai_flagship.rs` combines an exact-target Responses option, the portable language
  family, and the provider-owned Conversations lifecycle;
- `examples/anthropic_flagship.rs` combines Messages options, scope-bound Files-in-Messages,
  assistant-history replay, and a provider-owned Skills metadata list;
- `examples/trusted_connect_route.rs` constructs a trusted route through curated facade APIs and
  proves how a provider builder consumes the resulting settings without credentials or I/O.

The flagship examples are offline by default and perform network calls only after their provider
credential environment variable is set. The trusted-route example performs no network I/O.

See the [repository README](https://github.com/YumchaLabs/siumai#readme),
[migration guide](https://github.com/YumchaLabs/siumai/blob/main/docs/migration/siumai-next.md), and
[provider support policy](https://github.com/YumchaLabs/siumai/blob/main/docs/providers/support-policy.md)
for the primary user journey and current support evidence.
