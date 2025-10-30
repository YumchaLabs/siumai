# Extensibility Examples

This directory contains examples demonstrating how to extend Siumai with custom functionality.

## Examples

### 1. Custom Provider Parameters

File: `custom_provider_parameters.rs`

Description:
Demonstrates how to add new parameters to providers without waiting for library updates. Use new provider features immediately when they're released.

Use Cases:
- Provider releases a new feature before the library update
- Testing experimental/beta features
- Using private/custom provider extensions

Run:
```bash
cargo run --example custom_provider_parameters --features xai
```

Key Concepts:
1. Implement `CustomProviderOptions` trait
2. Use `ProviderOptions::from_custom()` to convert
3. Parameters are automatically injected into API requests via provider hooks
4. All supported providers work with this pattern

---

### 2. Custom Provider Implementation

File: `custom_provider_implementation.rs`

Description:
Demonstrates how to implement a completely custom AI provider. Suitable for private APIs, non-OpenAI-compatible APIs, etc.

Use Cases:
- Private/internal AI services
- Non-OpenAI-compatible APIs
- Custom protocol AI services
- Local model services

Run:
```bash
cargo run --example custom_provider_implementation
```

Key Concepts:
1. Implement the `CustomProvider` trait
2. Define supported models and capabilities
3. Implement `chat()` and `chat_stream()`
4. Full control over request and response formats

---

### 3. Custom Provider Options

File: `custom-provider-options.rs`

Description:
Demonstrates how to add support for new provider features using the `CustomProviderOptions` trait.

Use Cases:
- Adding support for new provider features before they're built into the library
- Experimenting with beta/preview features
- Supporting custom/proprietary provider extensions

Run:
```bash
cargo run --example custom-provider-options --features xai
```

Key Concepts:
1. Implement `CustomProviderOptions` for your custom features
2. Use `ProviderOptions::from_custom()` to convert to `ProviderOptions`
3. Custom options are injected by `ProviderSpec::chat_before_send()`
4. Only use the `Custom` variant for features not yet in the library
5. Prefer built-in type-safe options when available

## When to Use Custom Options

### Good Use Cases

- New features not yet supported by the library
- Beta/experimental features
- Proprietary or custom provider extensions
- Rapid prototyping without waiting for library updates

### Avoid Using Custom Options When

- A type-safe built-in option already exists
- Standard parameters are needed (use `CommonParams`)
- Provider-specific type-safe options are available (e.g., `OpenAiOptions`, `XaiOptions`)

## Migration Path

When a custom feature becomes officially supported:

Before (Custom):
```rust
let custom_feature = CustomXaiFeature {
    search_mode: Some("on".to_string()),
};

let req = ChatRequest::new(messages)
    .with_provider_options(ProviderOptions::from_custom(custom_feature)?);
```

After (Built-in):
```rust
let req = ChatRequest::new(messages)
    .with_xai_options(
        XaiOptions::new()
            .with_search(XaiSearchParameters {
                mode: SearchMode::On,
                ..Default::default()
            })
    );
```

## Best Practices

1. Prefer built-in type-safe options when available
2. Document which custom features you rely on
3. Track provider API versions for compatibility
4. Handle errors gracefully when providers reject custom features
5. Migrate to built-in options when they become available

## Contributing

If you frequently use the same custom features:
1. Open an issue requesting built-in support
2. Contribute a PR to add the feature
3. Share your `CustomProviderOptions` implementation with the community

---

### 4. Custom ProviderSpec

File: `custom_provider_spec.rs`

Description:
Demonstrates how to implement the `ProviderSpec` trait to create custom provider behavior, such as adding prompt prefixes or custom authentication.

Use Cases:
- Adding prompt prefixes to all requests
- Custom authentication mechanisms
- Modifying API responses
- Wrapping existing providers

Run:
```bash
cargo run --example custom_provider_spec --features openai
```

Key Concepts:
1. Implement the `ProviderSpec` trait
2. Create custom request/response transformers
3. Reuse existing transformers as a base
4. Use with `HttpChatExecutor`

---

### 5. Testing Executors

File: `testing_executors.rs`

Description:
Demonstrates how to test executors using mock `ProviderSpec` implementations without making real API calls.

Use Cases:
- Unit testing without API calls
- Testing error handling
- Simulating different scenarios
- CI/CD testing

Run:
```bash
cargo run --example testing_executors
```

Key Concepts:
1. Create mock `ProviderSpec`
2. Track call counts
3. Simulate errors
4. Use `wiremock` for integration tests

---

### 6. Complete Custom Provider

File: `complete_custom_provider.rs`

Description:
Shows a complete implementation of a custom provider with configuration, transformers, and client wrapper.

Use Cases:
- Adding support for new LLM providers
- Understanding the full provider pattern
- Creating provider wrappers

Run:
```bash
cargo run --example complete_custom_provider --features openai
```

Implementation Steps:
1. Define provider configuration
2. Implement the `ProviderSpec` trait
3. Create a request transformer
4. Create a response transformer
5. Build a client with the builder pattern

---

## Architecture Overview

### The ProviderSpec Pattern

```
┌─────────────────────────────────────────────────────────────┐
│                      HttpChatExecutor                        │
│  (Unified executor for all providers)                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ uses
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      ProviderSpec Trait                      │
│  - id() → provider identifier                               │
│  - build_headers() → HTTP headers                            │
│  - chat_url() → API endpoint                                 │
│  - choose_chat_transformers() → transformers                 │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ implemented by
                            ▼
┌──────────────┬──────────────┬──────────────┬──────────────┐
│  OpenAiSpec  │AnthropicSpec │  GeminiSpec  │ CustomSpec   │
│              │              │              │              │
│ OpenAI API   │Anthropic API │ Gemini API   │ Your API     │
└──────────────┴──────────────┴──────────────┴──────────────┘
```

### Key Components

1. **ProviderSpec**: Defines provider-specific behavior
2. **ProviderContext**: Holds provider configuration
3. **Transformers**: Convert between Siumai types and provider API formats
4. **Executor**: Handles HTTP communication

---

## Best Practices

### When Creating a Custom Provider

1. **Reuse existing transformers** when possible
2. **Implement proper error handling**
3. **Use builder pattern for configuration**
4. **Write tests with mock specs**

### When Testing

1. **Use mock specs for unit tests** (no HTTP calls)
2. **Use wiremock for integration tests** (mock HTTP server)
3. **Track call counts** to verify behavior
4. **Test error scenarios** explicitly

---

## Related Documentation

- [Provider Spec Refactor](../../docs/architecture/provider-spec-refactor.md)
- [Type-Safe Provider Options](../../siumai/src/types/provider_options.rs)
- [ProviderSpec Trait](../../siumai/src/provider_spec.rs)
- [Executor Refactoring](../../docs/refactoring/PHASE7_EXECUTOR_REFACTORING_COMPLETE.md)
- [Architecture Design](../../docs/architecture/capability-standards-design.md)
