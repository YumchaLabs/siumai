# Extensibility Examples / 扩展性示例

This directory contains examples demonstrating how to extend Siumai with custom functionality.

本目录包含演示如何使用自定义功能扩展Siumai的示例。

## Examples / 示例

### 1. Custom Provider Parameters / 自定义提供商参数

**File / 文件**: `custom_provider_parameters.rs`

**Description / 说明**:
Demonstrates how to add new parameters to providers without waiting for library updates. Use new provider features immediately when they're released.

演示如何为提供商添加新参数，无需等待库更新。当提供商发布新功能时，您可以立即使用。

**Use Cases / 使用场景**:
- ✅ Provider releases new feature before library update / 提供商发布了新功能，但库还未更新
- ✅ Testing experimental/beta features / 测试实验性/Beta功能
- ✅ Using private/custom provider extensions / 使用私有/定制的提供商扩展

**Run / 运行**:
```bash
cargo run --example custom_provider_parameters --features xai
```

**关键概念 / Key Concepts**:
1. 实现 `CustomProviderOptions` trait / Implement `CustomProviderOptions` trait
2. 使用 `ProviderOptions::from_custom()` 转换 / Use `ProviderOptions::from_custom()` to convert
3. 参数自动注入到API请求 / Parameters automatically injected into API requests
4. 所有6个提供商都支持 / All 6 providers support this feature

---

### 2. Custom Provider Implementation / 自定义提供商实现

**File / 文件**: `custom_provider_implementation.rs`

**Description / 说明**:
Demonstrates how to implement a completely custom AI provider. Suitable for private APIs, non-OpenAI-compatible APIs, etc.

演示如何实现一个完全自定义的AI提供商。适用于私有API、非OpenAI兼容的API等场景。

**Use Cases / 使用场景**:
- ✅ Private/internal AI services / 私有/内部AI服务
- ✅ Non-OpenAI-compatible APIs / 非OpenAI兼容的API
- ✅ Custom protocol AI services / 自定义协议的AI服务
- ✅ Local model services / 本地模型服务

**Run / 运行**:
```bash
cargo run --example custom_provider_implementation
```

**关键概念 / Key Concepts**:
1. 实现 `CustomProvider` trait / Implement `CustomProvider` trait
2. 定义支持的模型和能力 / Define supported models and capabilities
3. 实现 `chat()` 和 `chat_stream()` / Implement `chat()` and `chat_stream()`
4. 完全控制请求和响应格式 / Full control over request and response format

---

### 3. Custom Provider Options (English Version)

**File**: `custom-provider-options.rs`

**Description**:
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

