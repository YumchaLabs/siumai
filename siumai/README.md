# `siumai` facade

`siumai` is the curated public facade for the Siumai workspace. It re-exports six small
provider-neutral model-family contracts and opt-in provider-owned APIs without flattening native
resources or protocol controls into a least-common-denominator client.

The facade has no provider enabled by default. Select only the providers and orchestration layers
the application uses:

```toml
[dependencies]
siumai = { version = "0.11.0-beta.9", default-features = false, features = ["openai"] }
```

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

See the [repository README](https://github.com/YumchaLabs/siumai#readme),
[migration guide](https://github.com/YumchaLabs/siumai/blob/main/docs/migration/siumai-next.md), and
[provider support policy](https://github.com/YumchaLabs/siumai/blob/main/docs/providers/support-policy.md)
for the complete user journey and current support evidence.
