# ProviderOptions 标准化设计（v0.12+ 草案）

状态：草案（部分已在 0.11.x 实现）
更新时间：2025-11-16

本稿约束 `ProviderOptions` 的使用方式，并约定如何在
core / std / provider / 聚合 四层之间传递与生效。

## 1. 总体目标

- typed options：所有 provider 特有参数都通过类型安全的
  `ProviderOptions::<Provider>` 携带，避免裸 `HashMap<String, Value>`。
- 单一入口：聚合层只负责把 typed options 映射到
  `siumai-core::execution::chat::ChatInput::extra`。
- 单一出口：std/provider 层唯一负责把这些 extra 注入到
  最终 HTTP JSON body 中（request/stream），聚合层不再拆 JSON。
- 明确命名：`ChatInput::extra` 中的 key 采用
  `"<provider>_<feature>"` 或 `"<provider>_<group>_<field>"` 形式，
  避免 key 冲突。

## 2. 数据流约定

以 Chat 为例，完整数据流如下：

```text
ChatRequest
  └─ provider_options: ProviderOptions::<Provider>
      ↓（聚合层映射）
ChatInput
  └─ extra: Map<String, Value>
      ↓（std/provider adapter）
HTTP JSON body（provider 原生 schema）
```

各层职责：

- 聚合层（siumai crate）：
  - 在 `ProviderSpec::<Provider>::choose_chat_transformers` 中，将
    `ChatRequest::provider_options` 解析为 provider-specific 结构，
    并填入 `ChatInput::extra`（符合命名约定）。
  - 不直接修改 HTTP JSON body（除通用的 `CustomProviderOptions`）。
- core 标准层（siumai-std-*）：
  - 负责把 `ChatInput` 映射为 provider 标准 JSON 结构。
  - 通过 adapter trait（如 `OpenAiChatAdapter`）读取 `extra`，
    并在 `transform_request` 中注入 provider 特有字段。
- provider crate（siumai-provider-*）：
  - 实现 `CoreProviderSpec`，决定何时/如何调用 std 标准层。
  - 可以选择性提供额外的 adapter 实现，但不直接依赖聚合层类型。

## 3. 命名规范（ChatInput::extra）

### 3.1 Key 命名

- 统一前缀：使用 provider 名作为前缀，例如：
  - xAI: `xai_search_parameters`, `xai_reasoning_effort`
  - Groq: `groq_extra_params`
  - OpenAI: `openai_response_format`, `openai_service_tier`
- 分组字段：当某个 typed options 映射为完整对象时，可以使用
  `"<provider>_<group>"` 作为 key，例如：
  - `xai_search_parameters`（整个搜索配置对象）
  - `groq_extra_params`（整包原始 KV）

### 3.2 值约定

- 只存放已经序列化好的 `serde_json::Value`，不在 adapter 内再解析
  复杂 Rust 结构。
- 需要结构化配置时，类型定义应放在 `types::provider_options::<provider>`
  模块中，聚合层负责 `to_value`：

```rust
// 聚合层（示意）
if let ProviderOptions::Xai(ref options) = req.provider_options {
    if let Some(ref sp) = options.search_parameters
        && let Ok(v) = serde_json::to_value(sp)
    {
        input.extra.insert("xai_search_parameters".into(), v);
    }
}
```

## 4. 具体示例

### 4.1 xAI（Grok）

- typed options：`types::XaiOptions` + `XaiSearchParameters`。
- ChatRequest：

```rust,ignore
let req = ChatRequest::builder()
    .messages(..)
    .xai_options(
        XaiOptions::new()
            .with_search(XaiSearchParameters::default())
            .with_reasoning_effort("medium")
    )
    .build();
```

- 聚合层映射（`XaiSpec::choose_chat_transformers`）：

```rust,ignore
fn xai_chat_request_to_core_input(req: &ChatRequest) -> ChatInput {
    let mut input = openai_like_chat_request_to_core_input(req);

    if let ProviderOptions::Xai(ref options) = req.provider_options {
        if let Some(ref sp) = options.search_parameters
            && let Ok(v) = serde_json::to_value(sp)
        {
            input.extra.insert("xai_search_parameters".into(), v);
        }
        if let Some(ref effort) = options.reasoning_effort {
            input.extra.insert("xai_reasoning_effort".into(), Value::String(effort.clone()));
        }
    }

    input
}
```

- std/provider adapter（在 `siumai-provider-xai` 内实现）：

```rust,ignore
impl OpenAiChatAdapter for XaiOpenAiChatAdapter {
    fn transform_request(&self, input: &ChatInput, body: &mut Value) -> Result<(), LlmError> {
        if let Some(params) = input.extra.get("xai_search_parameters") {
            body["search_parameters"] = params.clone();
        }
        if let Some(effort) = input.extra.get("xai_reasoning_effort") {
            body["reasoning_effort"] = effort.clone();
        }
        Ok(())
    }
}
```

> 结果：HTTP JSON 中的 `search_parameters` / `reasoning_effort`
> 完全由 std/provider adapter 注入，聚合层只处理 typed options→extra。

### 4.2 Groq

- typed options：`types::GroqOptions`，对齐 Vercel 的 `groqProviderOptions`：
  - `reasoning_effort: Option<String>`（"none" | "default" | "low" | "medium" | "high"）
  - `reasoning_format: Option<String>`（"hidden" | "raw" | "parsed"）
  - `parallel_tool_calls: Option<bool>`
  - `service_tier: Option<String>`（"on_demand" | "flex" | "auto"）
  - `extra_params: HashMap<String, Value>`（逃生口）

- ChatRequest：

```rust,ignore
let groq_opts = GroqOptions::new()
    .with_reasoning_effort("medium")
    .with_reasoning_format("parsed")
    .with_parallel_tool_calls(true)
    .with_service_tier("flex")
    .with_param("custom_flag", json!(true));

let req = ChatRequest::builder()
    .messages(..)
    .with_groq_options(groq_opts)
    .build();
```

- 聚合层映射（`groq_chat_request_to_core_input` → `ChatInput::extra["groq_*"]`）：

```rust,ignore
pub fn groq_chat_request_to_core_input(req: &ChatRequest) -> ChatInput {
    let mut input: ChatInput = openai_like_chat_request_to_core_input(req);

    if let ProviderOptions::Groq(ref options) = req.provider_options {
        if let Some(ref effort) = options.reasoning_effort {
            input.extra.insert(
                "groq_reasoning_effort".to_string(),
                serde_json::json!(effort),
            );
        }
        if let Some(ref fmt) = options.reasoning_format {
            input
                .extra
                .insert("groq_reasoning_format".to_string(), serde_json::json!(fmt));
        }
        if let Some(parallel) = options.parallel_tool_calls {
            input.extra.insert(
                "groq_parallel_tool_calls".to_string(),
                serde_json::json!(parallel),
            );
        }
        if let Some(ref tier) = options.service_tier {
            input
                .extra
                .insert("groq_service_tier".to_string(), serde_json::json!(tier));
        }

        if !options.extra_params.is_empty()
            && let Ok(v) = serde_json::to_value(&options.extra_params)
        {
            input.extra.insert("groq_extra_params".to_string(), v);
        }
    }

    input
}
```

- std/provider adapter（`siumai-provider-groq::GroqOpenAiChatAdapter`）：

```rust,ignore
impl OpenAiChatAdapter for GroqOpenAiChatAdapter {
    fn transform_request(&self, input: &ChatInput, body: &mut Value) -> Result<(), LlmError> {
        if let Some(body_obj) = body.as_object_mut() {
            // 先展开逃生口 extra_params：仅在目标字段不存在时写入，避免覆盖 typed 字段。
            if let Some(extra) = input.extra.get("groq_extra_params")
                && let Some(extra_obj) = extra.as_object()
            {
                for (k, v) in extra_obj {
                    body_obj.entry(k.clone()).or_insert_with(|| v.clone());
                }
            }

            // 再映射 typed GroqOptions 字段（优先级高于 extra_params）。
            if let Some(effort) = input.extra.get("groq_reasoning_effort") {
                body_obj.insert("reasoning_effort".to_string(), effort.clone());
            }
            if let Some(fmt) = input.extra.get("groq_reasoning_format") {
                body_obj.insert("reasoning_format".to_string(), fmt.clone());
            }
            if let Some(parallel) = input.extra.get("groq_parallel_tool_calls") {
                body_obj.insert("parallel_tool_calls".to_string(), parallel.clone());
            }
            if let Some(tier) = input.extra.get("groq_service_tier") {
                body_obj.insert("service_tier".to_string(), tier.clone());
            }
        }
        Ok(())
    }
}
```

> 结果：GroqOptions 的 typed 字段（reasoning_effort / reasoning_format /
> parallel_tool_calls / service_tier）总是映射为请求体顶层字段，`extra_params`
> 仅在这些字段未显式设置时补充其他自定义参数，并且这一行为对聚合层透明。

### 4.3 OpenAI-Compatible（Custom ProviderOptions）

- typed options：OpenAI-Compatible 不增加新的 `ProviderOptions` 枚举分支，而是通过
  unified builder 将 provider-specific 配置写入 `ProviderOptions::Custom` 的 `options`
  字段，在 core 层统一落到 `ChatInput::extra["openai_compat_provider_params"]`。

- ChatRequest（以 unified builder 为例）：

```rust,ignore
let client = LlmBuilder::new()
    .openrouter()
    .api_key("your-key")
    .model("openai/gpt-4.1")
    .text_verbosity("low")
    .include_stream_usage(false)
    .build()
    .await?;
```

`OpenAiCompatibleBuilder` 内部会构造：

```rust,ignore
ProviderOptions::Custom {
    provider_id: "openrouter".into(),
    options: {
        "verbosity": "low",
        "include_stream_usage": false,
        // 其他 provider-specific 字段…
    }
}
```

- 聚合层映射（`OpenAiCompatibleSpec::choose_chat_transformers` 中的 `to_core` 闭包）：

```rust,ignore
let to_core = move |req: &crate::types::ChatRequest| {
    use crate::types::ProviderOptions;
    let mut input = crate::core::provider_spec::openai_chat_request_to_core_input(req);

    if let ProviderOptions::Custom { provider_id: custom_id, options } = &req.provider_options
        && *custom_id == provider_id_for_extra
        && !options.is_empty()
        && let Ok(v) = serde_json::to_value(options)
    {
        input
            .extra
            .insert("openai_compat_provider_params".to_string(), v);
    }

    input
};
```

- std/provider adapter（`CompatToOpenAiChatAdapter`，在启用 `std-openai-external` 时生效）：

```rust,ignore
impl OpenAiChatAdapter for CompatToOpenAiChatAdapter {
    fn transform_request(
        &self,
        req: &siumai_core::execution::chat::ChatInput,
        body: &mut Value,
    ) -> Result<(), LlmError> {
        // 展开 provider_params，但不覆盖 std-openai 已经设置的字段
        if let Some(extra) = req.extra.get("openai_compat_provider_params")
            && let Some(obj) = extra.as_object()
            && let Some(body_obj) = body.as_object_mut()
        {
            for (k, v) in obj {
                body_obj.entry(k.clone()).or_insert_with(|| v.clone());
            }
        }

        // Streaming usage 控制：include_usage 开关与 Vercel `includeUsage` 对齐
        if self.0.compatibility().supports_stream_options {
            let is_streaming = body
                .get("stream")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);

            if is_streaming {
                // builder 中的 include_stream_usage 显式关闭时，不注入 include_usage
                let include_flag = req
                    .extra
                    .get("openai_compat_provider_params")
                    .and_then(|v| v.get("include_stream_usage"))
                    .and_then(|v| v.as_bool());

                if include_flag.unwrap_or(true) {
                    if let Some(obj) = body.as_object_mut() {
                        let stream_options = obj
                            .entry("stream_options".to_string())
                            .or_insert_with(|| json!({}));
                        if let Some(so) = stream_options.as_object_mut() {
                            so.entry("include_usage".to_string())
                                .or_insert(json!(true));
                        }
                    }
                }
            }
        }

        self.0.transform_request_params(
            body,
            req.model.as_deref().unwrap_or(""),
            crate::providers::openai_compatible::types::RequestType::Chat,
        )
    }
}
```

> 结果：OpenAI-Compatible 的 provider-specific 配置经由
> `ProviderOptions::Custom` → `ChatInput::extra["openai_compat_provider_params"]`
> 统一传递，由 Compat adapter 在 std-openai 层展开到最终 JSON，且不会覆盖
> OpenAI 标准已经设置的字段；streaming usage 行为通过 `include_stream_usage`
> 开关与 Vercel 的 `includeUsage` 对齐。

### 4.4 Anthropic（Messages 标准）

- typed options：`types::AnthropicOptions`（含 `PromptCachingConfig`、`ThinkingModeConfig`、`AnthropicResponseFormat`）。
- ChatRequest：

```rust,ignore
let req = ChatRequest::builder()
    .messages(..)
    .anthropic_options(
        AnthropicOptions::new()
            .with_thinking_mode(ThinkingModeConfig::default())
            .with_json_object()
    )
    .build();
```

- 聚合层映射（`anthropic_like_chat_request_to_core_input`）：

```rust,ignore
pub fn anthropic_like_chat_request_to_core_input(
    req: &ChatRequest,
) -> ChatInput {
    use siumai_core::execution::chat::{ChatInput, ChatMessageInput, ChatRole};
    use std::collections::HashMap;

    let messages = /* 将 ChatRequest.messages 映射为 ChatMessageInput 列表 */;

    let mut extra: HashMap<String, Value> = HashMap::new();
    if let ProviderOptions::Anthropic(ref options) = req.provider_options {
        // Thinking 模式：在 extra 中构造最终协议 JSON 形状，
        // std 层只做轻量重命名。
        if let Some(ref thinking) = options.thinking_mode {
            if thinking.enabled {
                let mut thinking_config = json!({ "type": "enabled" });
                if let Some(budget) = thinking.thinking_budget {
                    thinking_config["budget_tokens"] = json!(budget);
                }
                extra.insert("anthropic_thinking".to_string(), thinking_config);
            }
        }

        // Structured output：JsonObject / JsonSchema
        if let Some(ref rf) = options.response_format {
            let value = match rf {
                AnthropicResponseFormat::JsonObject => {
                    json!({ "type": "json_object" })
                }
                AnthropicResponseFormat::JsonSchema {
                    name,
                    schema,
                    strict,
                } => json!({
                    "type": "json_schema",
                    "json_schema": {
                        "name": name,
                        "strict": strict,
                        "schema": schema,
                    }
                }),
            };
            extra.insert("anthropic_response_format".to_string(), value);
        }
    }

    ChatInput {
        messages,
        model: Some(req.common_params.model.clone()),
        max_tokens: req.common_params.max_tokens,
        temperature: req.common_params.temperature,
        top_p: req.common_params.top_p,
        presence_penalty: None,
        frequency_penalty: None,
        stop: req.common_params.stop_sequences.clone(),
        extra,
    }
}
```

- std/provider adapter（在 `siumai-std-anthropic` 内默认适配器实现）：

```rust,ignore
impl AnthropicChatAdapter for AnthropicDefaultChatAdapter {
    fn transform_request(
        &self,
        _input: &ChatInput,
        body: &mut Value,
    ) -> Result<(), LlmError> {
        if let Some(obj) = body.as_object_mut() {
            if let Some(thinking) = obj.remove("anthropic_thinking") {
                obj.insert("thinking".to_string(), thinking);
            }
            if let Some(response_format) = obj.remove("anthropic_response_format") {
                obj.insert("response_format".to_string(), response_format);
            }
        }
        Ok(())
    }
}
```

> 结果：Anthropic 的 thinking / response_format JSON 完全在
> `ChatInput::extra` → std adapter 这一链路中完成，聚合层只负责
> typed options → extra 的映射；`chat_before_send` 不再处理这些字段。

### 4.4 OpenAI-Compatible（兼容层）

- typed options：统一通过 `ProviderOptions::Custom` + OpenAI 标准 pipeline：
  - OpenAI-Compatible builder 将 provider-specific 配置（如 `verbosity` / `include_stream_usage` 等）
    写入 `provider_specific_config: HashMap<String, Value>`；
  - 在聚合层，`OpenAiCompatibleSpec` 的自定义 `ChatRequest -> ChatInput` 映射会把
    这些 KV 序列化后存入 `ChatInput::extra["openai_compat_provider_params"]`。

- ChatRequest（示例，以 OpenRouter 为例）：

```rust,ignore
let client = LlmBuilder::new()
    .openrouter()
    .api_key("your-api-key")
    .model("gpt-5.1-mini")
    .text_verbosity("low")              // → "verbosity": "low"
    .include_stream_usage(false)        // → 控制 stream_options.include_usage
    .build()
    .await?;
```

- 聚合层映射（`OpenAiCompatibleSpec` 内闭包 `to_core`）：

```rust,ignore
let to_core = move |req: &crate::types::ChatRequest| {
    use crate::types::ProviderOptions;
    let mut input = crate::core::provider_spec::openai_chat_request_to_core_input(req);

    if let ProviderOptions::Custom { provider_id, options } = &req.provider_options
        && *provider_id == provider_id_for_extra
        && !options.is_empty()
        && let Ok(v) = serde_json::to_value(options)
    {
        input
            .extra
            .insert("openai_compat_provider_params".to_string(), v);
    }

    input
};
```

- std/provider adapter（`CompatToOpenAiChatAdapter` 在 `std-openai-external` 模式下）：

```rust,ignore
impl OpenAiChatAdapter for CompatToOpenAiChatAdapter {
    fn transform_request(&self, req: &ChatInput, body: &mut Value) -> Result<(), LlmError> {
        // 将 extra["openai_compat_provider_params"] 扁平展开到最终 JSON，
        // 同名字段不覆盖 std-openai 已设置的值。
        if let Some(extra) = req.extra.get("openai_compat_provider_params")
            && let Some(obj) = extra.as_object()
            && let Some(body_obj) = body.as_object_mut()
        {
            for (k, v) in obj {
                body_obj.entry(k.clone()).or_insert_with(|| v.clone());
            }
        }

        // streaming usage 控制：当目标 provider 支持 stream_options 时，
        // 根据 include_stream_usage 决定是否注入 stream_options.include_usage = true。
        if self.0.compatibility().supports_stream_options {
            let is_streaming = body.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);
            if is_streaming {
                let include_flag = req
                    .extra
                    .get("openai_compat_provider_params")
                    .and_then(|v| v.get("include_stream_usage"))
                    .and_then(|v| v.as_bool());

                if include_flag.unwrap_or(true) {
                    if let Some(obj) = body.as_object_mut() {
                        let stream_options = obj
                            .entry("stream_options".to_string())
                            .or_insert_with(|| serde_json::json!({}));
                        if let Some(so) = stream_options.as_object_mut() {
                            so.entry("include_usage".to_string())
                                .or_insert(serde_json::Value::Bool(true));
                        }
                    }
                }
            }
        }

        self.0.transform_request_params(
            body,
            req.model.as_deref().unwrap_or(""),
            crate::providers::openai_compatible::types::RequestType::Chat,
        )
    }
}
```

> 结果：OpenAI-Compatible 作为“兼容层”本身不引入新的 typed ProviderOptions，
> 而是通过 `ProviderOptions::Custom` + `ChatInput::extra["openai_compat_provider_params"]`
> 把 provider-specific 字段交给 std-openai 标准 JSON 进行最后一层 merge，从而在不
> 污染聚合层的前提下，为各兼容 provider 提供 OpenAI 风格的参数入口。

## 5. OpenAI 示例（完成情况）

OpenAI 自身的 typed options（如 `OpenAiOptions::reasoning_effort`、
`service_tier`、`modalities`、`audio`、`prediction`、
`web_search_options` 等）已经按本标准完成收敛：

- Chat 路径：
  - 在 core 层提供 `openai_chat_request_to_core_input`，负责把
    `ProviderOptions::OpenAi` 映射到 `ChatInput.extra`：
    - `openai_reasoning_effort`
    - `openai_service_tier`
    - `openai_modalities`
    - `openai_audio`
    - `openai_prediction`
    - `openai_web_search_options`
  - 在 `siumai-std-openai` 中由 `OpenAiDefaultChatAdapter` 读取这些 extra
    key 并注入最终 JSON（仅在启用 `std-openai-external` 时生效）。
- Responses 路径：
  - 聚合层通过 `build_responses_input` 将 `ResponsesApiConfig` 以及
    OpenAI-specific 选项写入 `ResponsesInput::extra`；
  - `OpenAiResponsesStandard::transform_responses` 负责将 `extra` 展平
    到最终 `/responses` 请求体，行为等价于早期
    `OpenAiSpec::chat_before_send` 中的 JSON merge 逻辑。

在启用 `std-openai-external` 的场景下，聚合层不再直接基于
`ProviderOptions::OpenAi` 拼接 Chat/Responses JSON，只通过
`ChatInput::extra` / `ResponsesInput::extra` 传递 typed options。

## 6. 聚合层默认 hook（CustomProviderOptions）

`ProviderSpec::chat_before_send` 的默认实现
`default_custom_options_hook` 仍然保留，专门用于处理
`ProviderOptions::Custom`：

- typed options 统一走 `ChatInput::extra` → std/provider adapter。
- 真正完全自定义的 JSON merge（用户自写）继续走
  `ProviderOptions::Custom` + `chat_before_send` hook。

这两条路径互不干扰，typed options 优先，Custom 只做最后一层 merge。

## 7. 后续计划（0.12+ 优先级草案）

- OpenAI Chat legacy 路径收敛：
  - 在启用 `std-openai-external` 的场景下，仅通过 `ChatInput::extra` +
    `OpenAiDefaultChatAdapter` 注入 OpenAI-specific 字段；
  - 将聚合层 `chat_before_send` 中基于 `OpenAiOptions` 直接拼 JSON 的逻辑
    标记为 legacy，并在 0.12+ 版本中逐步移除。
- Web Search ProviderOptions：
  - Web 搜索不再作为独立 cross-provider 抽象能力；推荐通过各 provider 的
    provider-defined tools（如 `openai.web_search`）和 typed providerOptions
    使用搜索能力，聚合层不再维护统一的 WebSearch 参数映射逻辑。
- Ollama ProviderOptions 收敛：
  - 目前 `OllamaOptions` 在各自 capability 实现中直接转换为请求 JSON；
  - 如未来需要跨语言共享 Ollama 行为，可引入 `ollama_like_chat_request_to_core_input`
    / `ollama_like_embedding_request_to_core_input` + std/provider adapter，将
    typed options 统一落在 `extra["ollama_*"]`。

- MiniMaxi ProviderOptions 策略：
  - Chat 路径沿用 Anthropic Messages 标准，复用 `AnthropicOptions`：
    - typed 配置经由 `anthropic_like_chat_request_to_core_input` 写入
      `ChatInput::extra["anthropic_thinking" | "anthropic_response_format" | "anthropic_prompt_caching"]`；
    - MiniMaxi 标准与 provider crate 复用 Anthropic 标准的 JSON 形状，仅在 headers/URL/错误映射上区分。
  - 如未来 MiniMaxi 在 Chat 层新增明确的自有字段（非 Anthropic 兼容），再单独引入
    `MinimaxiOptions` + `minimaxi_chat_request_to_core_input`，typed 配置统一落在
    `ChatInput::extra["minimaxi_*"]`，避免与 Anthropic 的 key 混淆。

## 8. Usage 与 ProviderMetadata 规范（Usage & ProviderMetadata）

### 8.1 Usage 抽象统一规则

`crate::types::Usage` 是跨 provider 的统一用量抽象。各 provider 在填充 Usage 时需要遵守：

- 仅承载「token 相关」计数：`prompt_tokens` / `completion_tokens` / `total_tokens`。
- 更细粒度的拆分（如 reasoning / cached / prediction tokens）通过
  `prompt_tokens_details` / `completion_tokens_details` 表达。
- 非 token 型统计（如 queue_time / total_time / num_sources_used）不进入 Usage，
  如有需要应挂在 providerMetadata 或保留在原始 JSON 中。

对齐 OpenAI / Vercel 的具体约定：

- `prompt_tokens`：统一对应 input/prompt token 数：
  - OpenAI / Responses：`input_tokens | prompt_tokens | inputTokens`。
  - OpenAI-Compatible / xAI / Groq：`prompt_tokens`。
- `completion_tokens`：统一对应输出 token 数：
  - OpenAI / Responses：`output_tokens | completion_tokens | outputTokens`。
  - OpenAI-Compatible / xAI / Groq：`completion_tokens`。
- `total_tokens`：优先 provider 提供的 `total_tokens | totalTokens`，否则回退为
  `prompt_tokens + completion_tokens`。

扩展字段统一规则：

- `prompt_tokens_details.cached_tokens` → `Usage.prompt_tokens_details.cached_tokens`。
- `completion_tokens_details.reasoning_tokens` → `Usage.completion_tokens_details.reasoning_tokens`。
  - 对 OpenAI / Responses / OpenAI-Compatible / xAI：这是 reasoning 模型的标准位置。
  - 顶层 `usage.reasoning_tokens | reasoningTokens` 若存在，仅作为兼容入口：
    - 当 `completion_tokens_details.reasoning_tokens` 缺失时，才映射为 reasoning tokens。
- Predicted Outputs 相关：
  - `completion_tokens_details.accepted_prediction_tokens` →
    `Usage.completion_tokens_details.accepted_prediction_tokens`。
  - `completion_tokens_details.rejected_prediction_tokens` →
    `Usage.completion_tokens_details.rejected_prediction_tokens`。
- 已弃用字段 `Usage.cached_tokens` / `Usage.reasoning_tokens` 仍作为兼容出口：
  - builder 在构造 Usage 时会同步填充对应的 details 字段并保留旧字段，调用方应优先使用
    `prompt_tokens_details` / `completion_tokens_details`。

### 8.2 Provider 行为对齐快照

- OpenAI / OpenAI-Compatible / xAI：
  - 非流式：
    - OpenAI Chat / Responses：统一通过 std-openai 解析 usage，再在聚合层用 `Usage::builder`
      补充 reasoning / cached / prediction tokens。
    - OpenAI-Compatible：`CompatResponseTransformer` 直接读取 `usage` + details 字段，
      使用同一套 `Usage::builder` 规则。
    - xAI：`XaiResponseTransformer` 读取：
      - `prompt_tokens/completion_tokens/total_tokens`；
      - `prompt_tokens_details.cached_tokens`；
      - `completion_tokens_details.reasoning_tokens`（优先）；
      - 顶层 `reasoning_tokens` 仅作为 fallback。
  - 流式：
    - OpenAI / Responses：std-openai 的 stream converter 产生 `ChatStreamEventCore::UsageUpdate`，
      仅包含三项 token 计数；聚合层在需要时补充细节字段。
    - OpenAI-Compatible：`OpenAiCompatibleEventConverter` 在 streaming 过程中解析
      `usage.prompt_tokens_details.cached_tokens` 和
      `usage.completion_tokens_details.reasoning_tokens`，与非流式语义一致。
    - xAI：`XaiEventConverter` 在流式 usage 中使用与非流式同样的映射规则。
- Groq：
  - 根据 Groq 官方 schema，usage 包含 queue/prompt/completion/total 时间等字段；
    AI SDK 仅将 `prompt_tokens` / `completion_tokens` / `total_tokens` + `prompt_tokens_details.cached_tokens`
    映射到其 `usage` 抽象。
  - Siumai 中 `GroqUsage` 类型包含所有官方字段，但聚合层仅把 token 计数映射进 `Usage`，
    与 AI SDK 行为保持一致；queue_time 等延迟信息目前只保留在原始 JSON（未来如有需要可通过
    providerMetadata 暴露）。

### 8.3 ProviderMetadata（OpenAI / Anthropic / Gemini）

`ChatResponse.provider_metadata: Option<HashMap<String, HashMap<String, Value>>>` 用于承载
provider 特有的、非 token 计数类信息（包括某些 usage 派生指标），并提供类型安全的访问器：

- OpenAI：
  - 类型：`OpenAiMetadata`（`crate::types::OpenAiMetadata`）：
    - `reasoning_tokens: Option<u32>`：来自 `usage.completion_tokens_details.reasoning_tokens`，
      或顶层 `usage.reasoning_tokens | reasoningTokens`。
    - `system_fingerprint: Option<String>`：来自顶层 `system_fingerprint | systemFingerprint`。
    - `service_tier: Option<String>`：来自顶层 `service_tier | serviceTier`。
  - 访问方式：
    ```rust
    if let Some(meta) = response.openai_metadata() {
        meta.reasoning_tokens;
        meta.system_fingerprint;
        meta.service_tier;
    }
    ```
  - OpenAI Chat / Responses 在构造 `ChatResponse` 时，会在不破坏现有行为的前提下，将这些字段
    同步写入顶层 `response.system_fingerprint` / `response.service_tier`，并在
    `provider_metadata["openai"]` 下保留结构化元数据。
- Anthropic：
  - 类型：`AnthropicMetadata`：
    - 目前主要承载 prompt caching / thinking 等相关统计；具体字段随 std-anthropic 更新。
  - 访问方式：
    ```rust
    if let Some(meta) = response.anthropic_metadata() {
        meta.cache_read_input_tokens;
        meta.cache_creation_input_tokens;
    }
    ```
- Gemini：
  - 类型：`GeminiMetadata`：
    - 聚焦 grounding/file search 等能力暴露的元数据（如 `grounding_metadata`）。
  - 访问方式：
    ```rust
    if let Some(meta) = response.gemini_metadata() {
        if let Some(grounding) = &meta.grounding_metadata {
            // inspect grounding_support, sources, etc.
        }
    }
    ```

设计原则：

- Usage 只做「token 用量」的统一抽象；延迟/缓存命中率等高级指标放在 providerMetadata。
- providerMetadata 尽量贴近 provider 官方字段命名，不强行跨 provider 统一；通过 typed
  访问器隐藏 HashMap 细节。
- 不在 providerMetadata 中重复 Usage 已经提供的信息，避免二义性；仅在需要增强型视图时
  （如 OpenAI reasoning tokens）做少量重复暴露。
