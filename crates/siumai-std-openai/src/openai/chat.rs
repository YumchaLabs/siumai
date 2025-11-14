//! OpenAI Chat Completions Standard (external, minimal – no streaming)
//!
//! Converts core ChatInput/ChatResult to/from OpenAI's Chat Completions JSON.

use siumai_core::error::LlmError;
use siumai_core::execution::chat::{
    ChatInput, ChatRequestTransformer, ChatResponseTransformer, ChatResult, ChatRole, ChatUsage,
};
use std::sync::Arc;

#[derive(Clone)]
pub struct OpenAiChatStandard {
    adapter: Option<Arc<dyn OpenAiChatAdapter>>,
}

impl OpenAiChatStandard {
    pub fn new() -> Self {
        Self { adapter: None }
    }
    pub fn with_adapter(adapter: Arc<dyn OpenAiChatAdapter>) -> Self {
        Self {
            adapter: Some(adapter),
        }
    }

    pub fn create_request_transformer(&self, provider_id: &str) -> Arc<dyn ChatRequestTransformer> {
        Arc::new(OpenAiChatRequestTransformer {
            provider_id: provider_id.to_string(),
            adapter: self.adapter.clone(),
        })
    }
    pub fn create_response_transformer(
        &self,
        provider_id: &str,
    ) -> Arc<dyn ChatResponseTransformer> {
        Arc::new(OpenAiChatResponseTransformer {
            provider_id: provider_id.to_string(),
            adapter: self.adapter.clone(),
        })
    }
    pub fn create_stream_converter(
        &self,
        provider_id: &str,
    ) -> Arc<dyn siumai_core::execution::streaming::ChatStreamEventConverterCore> {
        Arc::new(OpenAiChatStreamConverter {
            provider_id: provider_id.to_string(),
            adapter: self.adapter.clone(),
        })
    }
}

impl Default for OpenAiChatStandard {
    fn default() -> Self {
        Self::new()
    }
}

/// Adapter for provider-specific diffs
pub trait OpenAiChatAdapter: Send + Sync {
    fn transform_request(
        &self,
        _req: &ChatInput,
        _body: &mut serde_json::Value,
    ) -> Result<(), LlmError> {
        Ok(())
    }
    fn transform_response(&self, _resp: &mut serde_json::Value) -> Result<(), LlmError> {
        Ok(())
    }
    fn chat_endpoint(&self) -> &str {
        "/chat/completions"
    }
    fn transform_sse_event(&self, _event: &mut serde_json::Value) -> Result<(), LlmError> {
        Ok(())
    }
}

#[derive(Clone)]
struct OpenAiChatStreamConverter {
    provider_id: String,
    adapter: Option<Arc<dyn OpenAiChatAdapter>>,
}

impl siumai_core::execution::streaming::ChatStreamEventConverterCore for OpenAiChatStreamConverter {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }
    fn convert_event(
        &self,
        event: eventsource_stream::Event,
    ) -> Vec<Result<siumai_core::execution::streaming::ChatStreamEventCore, LlmError>> {
        let mut out = Vec::new();
        let data = event.data;
        if data.trim() == "[DONE]" {
            return out;
        }
        let mut v: serde_json::Value = match serde_json::from_str(&data) {
            Ok(v) => v,
            Err(e) => {
                return vec![Err(LlmError::ParseError(format!(
                    "Invalid SSE JSON: {}",
                    e
                )))];
            }
        };
        if let Some(adapter) = &self.adapter {
            if let Err(e) = adapter.transform_sse_event(&mut v) {
                return vec![Err(e)];
            }
        }
        if let Some(choices) = v.get("choices").and_then(|c| c.as_array()) {
            for (i, ch) in choices.iter().enumerate() {
                if let Some(delta) = ch.get("delta") {
                    if let Some(text) = delta.get("content").and_then(|s| s.as_str()) {
                        out.push(Ok(
                            siumai_core::execution::streaming::ChatStreamEventCore::ContentDelta {
                                delta: text.to_string(),
                                index: Some(i),
                            },
                        ));
                    }
                    if let Some(tc_arr) = delta.get("tool_calls").and_then(|a| a.as_array()) {
                        for tc in tc_arr.iter() {
                            let id = tc.get("id").and_then(|s| s.as_str()).map(|s| s.to_string());
                            let func = tc.get("function").cloned().unwrap_or(serde_json::json!({}));
                            let name = func
                                .get("name")
                                .and_then(|s| s.as_str())
                                .map(|s| s.to_string());
                            let args = func
                                .get("arguments")
                                .and_then(|s| s.as_str())
                                .map(|s| s.to_string());
                            out.push(Ok(siumai_core::execution::streaming::ChatStreamEventCore::ToolCallDelta { id, function_name: name, arguments_delta: args, index: Some(i) }));
                        }
                    }
                }
            }
        }
        if let Some(usage) = v.get("usage") {
            let pt = usage
                .get("prompt_tokens")
                .and_then(|n| n.as_u64())
                .unwrap_or(0) as u32;
            let ct = usage
                .get("completion_tokens")
                .and_then(|n| n.as_u64())
                .unwrap_or(0) as u32;
            let tt = usage
                .get("total_tokens")
                .and_then(|n| n.as_u64())
                .unwrap_or(pt as u64 + ct as u64) as u32;
            out.push(Ok(
                siumai_core::execution::streaming::ChatStreamEventCore::UsageUpdate {
                    prompt_tokens: pt,
                    completion_tokens: ct,
                    total_tokens: tt,
                },
            ));
        }
        if out.is_empty() {
            out.push(Ok(
                siumai_core::execution::streaming::ChatStreamEventCore::Custom {
                    event_type: "openai:unknown_chunk".into(),
                    data: v,
                },
            ));
        }
        out
    }
}

#[derive(Clone)]
struct OpenAiChatRequestTransformer {
    provider_id: String,
    adapter: Option<Arc<dyn OpenAiChatAdapter>>,
}

impl ChatRequestTransformer for OpenAiChatRequestTransformer {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }
    fn transform_chat(&self, req: &ChatInput) -> Result<serde_json::Value, LlmError> {
        // Map messages
        let messages = req
            .messages
            .iter()
            .map(|m| match m.role {
                ChatRole::System => serde_json::json!({ "role": "system", "content": m.content }),
                ChatRole::User => serde_json::json!({ "role": "user", "content": m.content }),
                ChatRole::Assistant => {
                    serde_json::json!({ "role": "assistant", "content": m.content })
                }
            })
            .collect::<Vec<_>>();

        let mut body = serde_json::json!({ "messages": messages });
        if let Some(model) = &req.model {
            body["model"] = serde_json::json!(model);
        }
        if let Some(n) = req.max_tokens {
            body["max_tokens"] = serde_json::json!(n);
        }
        if let Some(t) = req.temperature {
            body["temperature"] = serde_json::json!(t);
        }
        if let Some(t) = req.top_p {
            body["top_p"] = serde_json::json!(t);
        }
        if let Some(p) = req.presence_penalty {
            body["presence_penalty"] = serde_json::json!(p);
        }
        if let Some(f) = req.frequency_penalty {
            body["frequency_penalty"] = serde_json::json!(f);
        }
        if let Some(stop) = &req.stop {
            body["stop"] = serde_json::json!(stop);
        }

        if let Some(obj) = body.as_object_mut() {
            for (k, v) in &req.extra {
                obj.insert(k.clone(), v.clone());
            }
        }

        if let Some(adapter) = &self.adapter {
            adapter.transform_request(req, &mut body)?;
        }
        Ok(body)
    }
}

#[derive(Clone)]
struct OpenAiChatResponseTransformer {
    provider_id: String,
    adapter: Option<Arc<dyn OpenAiChatAdapter>>,
}

impl ChatResponseTransformer for OpenAiChatResponseTransformer {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }
    fn transform_chat_response(&self, raw: &serde_json::Value) -> Result<ChatResult, LlmError> {
        let mut resp = raw.clone();
        if let Some(adapter) = &self.adapter {
            adapter.transform_response(&mut resp)?;
        }

        // Parse minimal OpenAI chat response
        let content = resp["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("")
            .to_string();
        let finish_reason = resp["choices"][0]["finish_reason"]
            .as_str()
            .map(|s| s.to_string());
        let usage = if let Some(u) = resp.get("usage") {
            Some(ChatUsage {
                prompt_tokens: u["prompt_tokens"].as_u64().unwrap_or(0) as u32,
                completion_tokens: u["completion_tokens"].as_u64().unwrap_or(0) as u32,
                total_tokens: u["total_tokens"].as_u64().unwrap_or(0) as u32,
            })
        } else {
            None
        };
        Ok(ChatResult {
            content,
            finish_reason,
            usage,
            metadata: Default::default(),
        })
    }
}
