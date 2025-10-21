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

---

### 4. Custom ProviderSpec / 自定义 ProviderSpec

**File / 文件**: `custom_provider_spec.rs`

**Description / 说明**:
Demonstrates how to implement the `ProviderSpec` trait to create custom provider behavior, such as adding prompt prefixes or custom authentication.

演示如何实现 `ProviderSpec` trait 来创建自定义提供商行为，例如添加提示前缀或自定义认证。

**Use Cases / 使用场景**:
- ✅ Adding prompt prefixes to all requests / 为所有请求添加提示前缀
- ✅ Custom authentication mechanisms / 自定义认证机制
- ✅ Modifying API responses / 修改API响应
- ✅ Wrapping existing providers / 包装现有提供商

**Run / 运行**:
```bash
cargo run --example custom_provider_spec --features openai
```

**Key Concepts / 关键概念**:
1. Implement `ProviderSpec` trait / 实现 `ProviderSpec` trait
2. Create custom request/response transformers / 创建自定义请求/响应转换器
3. Reuse existing transformers as base / 复用现有转换器作为基础
4. Use with `HttpChatExecutor` / 与 `HttpChatExecutor` 一起使用

---

### 5. Testing Executors / 测试 Executors

**File / 文件**: `testing_executors.rs`

**Description / 说明**:
Demonstrates how to test executors using mock `ProviderSpec` implementations without making real API calls.

演示如何使用模拟的 `ProviderSpec` 实现来测试 executors，无需进行真实的API调用。

**Use Cases / 使用场景**:
- ✅ Unit testing without API calls / 无需API调用的单元测试
- ✅ Testing error handling / 测试错误处理
- ✅ Simulating different scenarios / 模拟不同场景
- ✅ CI/CD testing / CI/CD测试

**Run / 运行**:
```bash
cargo run --example testing_executors
```

**Key Concepts / 关键概念**:
1. Create mock `ProviderSpec` / 创建模拟 `ProviderSpec`
2. Track call counts / 跟踪调用次数
3. Simulate errors / 模拟错误
4. Use with `wiremock` for integration tests / 使用 `wiremock` 进行集成测试

---

### 6. Complete Custom Provider / 完整的自定义提供商

**File / 文件**: `complete_custom_provider.rs`

**Description / 说明**:
Shows a complete implementation of a custom provider with configuration, transformers, and client wrapper.

展示一个完整的自定义提供商实现，包括配置、转换器和客户端包装器。

**Use Cases / 使用场景**:
- ✅ Adding support for new LLM providers / 添加新的LLM提供商支持
- ✅ Understanding full provider pattern / 理解完整的提供商模式
- ✅ Creating provider wrappers / 创建提供商包装器

**Run / 运行**:
```bash
cargo run --example complete_custom_provider --features openai
```

**Implementation Steps / 实现步骤**:
1. Define provider configuration / 定义提供商配置
2. Implement `ProviderSpec` trait / 实现 `ProviderSpec` trait
3. Create request transformer / 创建请求转换器
4. Create response transformer / 创建响应转换器
5. Build client with builder pattern / 使用构建器模式构建客户端

---

## Architecture Overview / 架构概览

### The ProviderSpec Pattern / ProviderSpec 模式

```
┌─────────────────────────────────────────────────────────────┐
│                      HttpChatExecutor                        │
│  (Unified executor for all providers)                       │
│  (所有提供商的统一执行器)                                      │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ uses / 使用
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      ProviderSpec Trait                      │
│  - id() → provider identifier / 提供商标识符                 │
│  - build_headers() → HTTP headers / HTTP头                  │
│  - chat_url() → API endpoint / API端点                      │
│  - choose_chat_transformers() → transformers / 转换器        │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ implemented by / 实现者
                            ▼
┌──────────────┬──────────────┬──────────────┬──────────────┐
│  OpenAiSpec  │AnthropicSpec │  GeminiSpec  │ CustomSpec   │
│              │              │              │              │
│ OpenAI API   │Anthropic API │ Gemini API   │ Your API     │
└──────────────┴──────────────┴──────────────┴──────────────┘
```

### Key Components / 关键组件

1. **ProviderSpec**: Defines provider-specific behavior / 定义提供商特定行为
2. **ProviderContext**: Holds provider configuration / 保存提供商配置
3. **Transformers**: Convert between Siumai types and provider API formats / 在Siumai类型和提供商API格式之间转换
4. **Executor**: Handles HTTP communication / 处理HTTP通信

---

## Best Practices / 最佳实践

### When Creating a Custom Provider / 创建自定义提供商时

1. **Reuse existing transformers** when possible / 尽可能复用现有转换器
2. **Implement proper error handling** / 实现适当的错误处理
3. **Use builder pattern for configuration** / 使用构建器模式进行配置
4. **Write tests with mock specs** / 使用模拟spec编写测试

### When Testing / 测试时

1. **Use mock specs for unit tests** (no HTTP calls) / 使用模拟spec进行单元测试（无HTTP调用）
2. **Use wiremock for integration tests** (mock HTTP server) / 使用wiremock进行集成测试（模拟HTTP服务器）
3. **Track call counts** to verify behavior / 跟踪调用次数以验证行为
4. **Test error scenarios** explicitly / 明确测试错误场景

---

## Related Documentation / 相关文档

- [Provider Spec Refactor](../../docs/architecture/provider-spec-refactor.md)
- [Type-Safe Provider Options](../../siumai/src/types/provider_options.rs)
- [ProviderSpec Trait](../../siumai/src/provider_spec.rs)
- [Executor Refactoring](../../docs/refactoring/PHASE7_EXECUTOR_REFACTORING_COMPLETE.md)
- [Architecture Design](../../docs/architecture/capability-standards-design.md)

