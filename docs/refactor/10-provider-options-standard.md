# ProviderOptions 标准化设计（v0.12+ 草案）

状态：草案（内部约定）
更新时间：2025-11-15

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

- typed options：`types::GroqOptions { extra_params: HashMap<String, Value> }`。
- ChatRequest：

```rust,ignore
let groq_opts = GroqOptions::new()
    .with_param("service_tier", json!("lite"))
    .with_param("some_flag", json!(true));

let req = ChatRequest::builder()
    .messages(..)
    .groq_options(groq_opts)
    .build();
```

- 聚合层映射（`GroqSpec::choose_chat_transformers`）：

```rust,ignore
fn groq_chat_request_to_core_input(req: &ChatRequest) -> ChatInput {
    let mut input = openai_like_chat_request_to_core_input(req);

    if let ProviderOptions::Groq(ref options) = req.provider_options
        && !options.extra_params.is_empty()
        && let Ok(v) = serde_json::to_value(&options.extra_params)
    {
        input.extra.insert("groq_extra_params".into(), v);
    }

    input
}
```

- std/provider adapter（在 `siumai-provider-groq` 内实现）：

```rust,ignore
impl OpenAiChatAdapter for GroqOpenAiChatAdapter {
    fn transform_request(&self, input: &ChatInput, body: &mut Value) -> Result<(), LlmError> {
        if let Some(extra) = input.extra.get("groq_extra_params")
            && let Some(extra_obj) = extra.as_object()
            && let Some(body_obj) = body.as_object_mut()
        {
            for (k, v) in extra_obj {
                body_obj.insert(k.clone(), v.clone());
            }
        }
        Ok(())
    }
}
```

> 结果：GroqOptions 中的任意 KV 最终都以原样 JSON 形式出现在请求 body 中，
> 且这一行为对聚合层透明。

### 4.3 Anthropic（Messages 标准）

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

## 5. OpenAI 示例（计划）

OpenAI 自身的 typed options（如 `OpenAiOptions::response_format`、
`service_tier` 等）按同样模式处理：

- 在聚合层 `OpenAiSpec` 侧提供 `openai_chat_request_to_core_input`，
  负责把 `ProviderOptions::OpenAi` 映射到 `ChatInput.extra`：
  - `openai_response_format`
  - `openai_service_tier`
  - …
- 在 `siumai-provider-openai` 中为 `OpenAiStandardAdapter` 增加
  对这些 extra key 的处理逻辑。

该重构尚未完成，但应遵循本文件的约定。

## 6. 聚合层默认 hook（CustomProviderOptions）

`ProviderSpec::chat_before_send` 的默认实现
`default_custom_options_hook` 仍然保留，专门用于处理
`ProviderOptions::Custom`：

- typed options 统一走 `ChatInput::extra` → std/provider adapter。
- 真正完全自定义的 JSON merge（用户自写）继续走
  `ProviderOptions::Custom` + `chat_before_send` hook。

这两条路径互不干扰，typed options 优先，Custom 只做最后一层 merge。
