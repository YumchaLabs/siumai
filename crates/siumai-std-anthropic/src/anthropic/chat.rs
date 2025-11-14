//! Anthropic Messages Chat Standard（初始版本）
//!
//! 目标：
//! - 提供 provider 无关的 Anthropic Chat API 标准映射（请求/响应/流式）
//! - 仅依赖 `siumai-core` 的抽象；具体 provider 通过适配器 trait 注入差异
//!
//! 当前实现：
//! - 仅定义结构与接口（`AnthropicChatStandard` + `AnthropicChatAdapter`），
//!   内部转换逻辑将逐步从聚合 crate 迁移过来。

use serde::Deserialize;
use siumai_core::error::LlmError;
use siumai_core::execution::chat::{
    ChatInput, ChatRequestTransformer, ChatResponseTransformer, ChatResult,
};
use siumai_core::execution::streaming::{ChatStreamEventConverterCore, ChatStreamEventCore};
use std::sync::Arc;

/// Anthropic Chat 标准入口
#[derive(Clone, Default)]
pub struct AnthropicChatStandard {
    adapter: Option<Arc<dyn AnthropicChatAdapter>>,
}

impl AnthropicChatStandard {
    pub fn new() -> Self {
        Self { adapter: None }
    }

    pub fn with_adapter(adapter: Arc<dyn AnthropicChatAdapter>) -> Self {
        Self {
            adapter: Some(adapter),
        }
    }

    /// 创建请求转换器（core ChatInput -> Anthropic JSON）
    pub fn create_request_transformer(&self, provider_id: &str) -> Arc<dyn ChatRequestTransformer> {
        Arc::new(AnthropicChatRequestTx {
            provider_id: provider_id.to_string(),
            adapter: self.adapter.clone(),
        })
    }

    /// 创建响应转换器（Anthropic JSON -> core ChatResult）
    pub fn create_response_transformer(
        &self,
        provider_id: &str,
    ) -> Arc<dyn ChatResponseTransformer> {
        Arc::new(AnthropicChatResponseTx {
            provider_id: provider_id.to_string(),
            adapter: self.adapter.clone(),
        })
    }

    /// 创建流式事件转换器（SSE -> ChatStreamEventCore）
    pub fn create_stream_converter(
        &self,
        provider_id: &str,
    ) -> Arc<dyn ChatStreamEventConverterCore> {
        Arc::new(AnthropicChatStreamConv {
            provider_id: provider_id.to_string(),
            adapter: self.adapter.clone(),
        })
    }
}

/// Anthropic Chat provider 适配器，用于注入少量差异
pub trait AnthropicChatAdapter: Send + Sync {
    /// 请求 JSON 调整（模型别名、特殊参数等）
    fn transform_request(
        &self,
        _input: &ChatInput,
        _body: &mut serde_json::Value,
    ) -> Result<(), LlmError> {
        Ok(())
    }

    /// 响应 JSON 调整（兼容不同版本字段）
    fn transform_response(&self, _resp: &mut serde_json::Value) -> Result<(), LlmError> {
        Ok(())
    }

    /// SSE 事件 JSON 调整
    fn transform_sse_event(&self, _event: &mut serde_json::Value) -> Result<(), LlmError> {
        Ok(())
    }

    /// Messages 端点路径（默认 `/v1/messages`）
    fn messages_endpoint(&self) -> &str {
        "/v1/messages"
    }
}

#[derive(Clone)]
struct AnthropicChatRequestTx {
    provider_id: String,
    adapter: Option<Arc<dyn AnthropicChatAdapter>>,
}

impl ChatRequestTransformer for AnthropicChatRequestTx {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }

    fn transform_chat(&self, input: &ChatInput) -> Result<serde_json::Value, LlmError> {
        if input.model.as_deref().unwrap_or("").is_empty() {
            return Err(LlmError::InvalidParameter("Model must be specified".into()));
        }

        // 使用 utils 构建 Anthropic Messages 结构 + system
        let (messages, system) = crate::anthropic::utils::build_messages_payload(input)?;

        let mut body = serde_json::json!({
            "model": input.model.clone().unwrap_or_default(),
            "messages": messages,
        });

        if let Some(sys) = system {
            body["system"] = serde_json::json!(sys);
        }
        if let Some(mt) = input.max_tokens {
            body["max_tokens"] = serde_json::json!(mt);
        }
        if let Some(t) = input.temperature {
            body["temperature"] = serde_json::json!(t);
        }
        if let Some(tp) = input.top_p {
            body["top_p"] = serde_json::json!(tp);
        }

        if let Some(obj) = body.as_object_mut() {
            for (k, v) in &input.extra {
                obj.insert(k.clone(), v.clone());
            }
        }

        if let Some(adapter) = &self.adapter {
            adapter.transform_request(input, &mut body)?;
        }
        Ok(body)
    }
}

#[derive(Clone)]
struct AnthropicChatResponseTx {
    provider_id: String,
    adapter: Option<Arc<dyn AnthropicChatAdapter>>,
}

impl ChatResponseTransformer for AnthropicChatResponseTx {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }

    fn transform_chat_response(&self, raw: &serde_json::Value) -> Result<ChatResult, LlmError> {
        let mut resp = raw.clone();
        if let Some(adapter) = &self.adapter {
            adapter.transform_response(&mut resp)?;
        }

        // 使用 utils 解析最小 ChatResult（后续可在 utils 中增强）
        Ok(crate::anthropic::utils::parse_minimal_chat_result(&resp))
    }
}

#[derive(Clone)]
struct AnthropicChatStreamConv {
    provider_id: String,
    adapter: Option<Arc<dyn AnthropicChatAdapter>>,
}

impl ChatStreamEventConverterCore for AnthropicChatStreamConv {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }

    fn convert_event(
        &self,
        event: eventsource_stream::Event,
    ) -> Vec<Result<ChatStreamEventCore, LlmError>> {
        #[derive(Debug, Clone, Deserialize, serde::Serialize)]
        struct AnthropicStreamEvent {
            #[serde(rename = "type")]
            kind: String,
            #[serde(default)]
            delta: Option<AnthropicDelta>,
            #[serde(default)]
            usage: Option<AnthropicUsage>,
        }

        #[derive(Debug, Clone, Deserialize, serde::Serialize)]
        struct AnthropicDelta {
            #[serde(default)]
            text: Option<String>,
            #[serde(default)]
            thinking: Option<String>,
        }

        #[derive(Debug, Clone, Deserialize, serde::Serialize)]
        struct AnthropicUsage {
            #[serde(default)]
            input_tokens: Option<u32>,
            #[serde(default)]
            output_tokens: Option<u32>,
        }

        let mut out = Vec::new();
        let data = event.data.trim();
        if data.is_empty() || data == "[DONE]" {
            return out;
        }

        let mut evt: AnthropicStreamEvent = match serde_json::from_str(data) {
            Ok(v) => v,
            Err(e) => {
                return vec![Err(LlmError::ParseError(format!(
                    "Anthropic SSE parse error: {}",
                    e
                )))];
            }
        };

        // 允许 adapter 修改底层 JSON（供代理/变体使用）
        if let Some(adapter) = &self.adapter {
            let mut raw = serde_json::to_value(&evt).unwrap_or_else(|_| serde_json::json!({}));
            if let Err(e) = adapter.transform_sse_event(&mut raw) {
                return vec![Err(e)];
            }
            // 尝试再解析回标准事件结构；失败则以 Custom 事件返回
            evt = match serde_json::from_value(raw.clone()) {
                Ok(v) => v,
                Err(_) => {
                    return vec![Ok(ChatStreamEventCore::Custom {
                        event_type: "anthropic:unknown_chunk".into(),
                        data: raw,
                    })];
                }
            };
        }

        match evt.kind.as_str() {
            "message_start" => {
                // 可以在此处发出 StreamStart，如后续需要
            }
            "message_delta" | "content_block_delta" | "message_delta_input_json_delta" => {
                if let Some(d) = evt.delta {
                    if let Some(text) = d.text {
                        if !text.is_empty() {
                            out.push(Ok(ChatStreamEventCore::ContentDelta {
                                delta: text,
                                index: None,
                            }));
                        }
                    }
                    if let Some(th) = d.thinking {
                        if !th.is_empty() {
                            out.push(Ok(ChatStreamEventCore::ThinkingDelta { delta: th }));
                        }
                    }
                }
            }
            "message_stop" | "message_end" => {
                if let Some(u) = evt.usage {
                    let prompt_tokens = u.input_tokens.unwrap_or(0);
                    let completion_tokens = u.output_tokens.unwrap_or(0);
                    let total_tokens = prompt_tokens.saturating_add(completion_tokens);
                    out.push(Ok(ChatStreamEventCore::UsageUpdate {
                        prompt_tokens,
                        completion_tokens,
                        total_tokens,
                    }));
                }
            }
            _ => {}
        }

        if out.is_empty() {
            // 将无法识别的事件以 Custom 形式抛出，便于上层调试
            let raw: serde_json::Value =
                serde_json::from_str(data).unwrap_or_else(|_| serde_json::json!({}));
            out.push(Ok(ChatStreamEventCore::Custom {
                event_type: "anthropic:unknown_chunk".into(),
                data: raw,
            }));
        }
        out
    }
}
