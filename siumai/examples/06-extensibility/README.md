# Extensibility Examples

This directory contains examples demonstrating how to extend Siumai with custom functionality.

## Examples

### Custom Provider Options

**File**: `custom-provider-options.rs`

Demonstrates how to add support for new provider features using the `CustomProviderOptions` trait.

**Use Cases**:
- Adding support for new provider features before they're built into the library
- Experimenting with beta/preview features
- Supporting custom/proprietary provider extensions

**Run**:
```bash
cargo run --example custom-provider-options --features xai
```

**Key Concepts**:
1. Implement `CustomProviderOptions` trait for your custom features
2. Use `ProviderOptions::from_custom()` to convert to `ProviderOptions`
3. Custom options are automatically injected by `ProviderSpec::chat_before_send()`
4. Only use `Custom` variant for features not yet in the library
5. Prefer built-in type-safe options when available

## When to Use Custom Options

### ✅ Good Use Cases

- **New Features**: Provider releases a new feature that isn't in the library yet
- **Beta Features**: Testing experimental features before official support
- **Custom Providers**: Supporting proprietary or custom provider extensions
- **Rapid Prototyping**: Quick experimentation without waiting for library updates

### ❌ Bad Use Cases

- **Built-in Features**: Don't use Custom for features that already have type-safe support
- **Standard Parameters**: Use `CommonParams` for standard parameters like `temperature`, `max_tokens`
- **Provider-Specific Options**: Use built-in options like `OpenAiOptions`, `XaiOptions`, etc.

## Migration Path

When a custom feature becomes officially supported:

**Before** (Custom):
```rust
let custom_feature = CustomXaiFeature {
    search_mode: Some("on".to_string()),
};

let req = ChatRequest::new(messages)
    .with_provider_options(ProviderOptions::from_custom(custom_feature)?);
```

**After** (Built-in):
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

1. **Type Safety First**: Always prefer built-in type-safe options when available
2. **Document Your Extensions**: Clearly document what custom features you're using
3. **Version Awareness**: Track which provider API version your custom features target
4. **Error Handling**: Custom features may fail if the provider doesn't support them
5. **Migration Plan**: Plan to migrate to built-in options when they become available

## Contributing

If you find yourself using custom options frequently for a particular feature, consider:
1. Opening an issue to request built-in support
2. Contributing a PR to add the feature to the library
3. Sharing your `CustomProviderOptions` implementation with the community

## Related Documentation

- [Provider Spec Refactor](../../docs/architecture/provider-spec-refactor.md)
- [Type-Safe Provider Options](../../siumai/src/types/provider_options.rs)
- [ProviderSpec Trait](../../siumai/src/provider_spec.rs)

