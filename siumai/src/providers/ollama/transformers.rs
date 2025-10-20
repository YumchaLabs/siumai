//! Transformers for Ollama Chat
//!
//! Request/Response transformers wiring to enable HttpChatExecutor path.

use crate::error::LlmError;
use crate::transformers::{request::RequestTransformer, response::ResponseTransformer};
use crate::types::{ChatRequest, ChatResponse, FinishReason, Usage};

#[derive(Clone)]
pub struct OllamaRequestTransformer {
    pub params: super::config::OllamaParams,
}

impl RequestTransformer for OllamaRequestTransformer {
    fn provider_id(&self) -> &str {
        "ollama"
    }

    fn transform_chat(&self, req: &ChatRequest) -> Result<serde_json::Value, LlmError> {
        if req.common_params.model.is_empty() {
            return Err(LlmError::InvalidParameter(
                "Model must be specified".to_string(),
            ));
        }

        // Messages
        let messages: Vec<super::types::OllamaChatMessage> = req
            .messages
            .iter()
            .map(super::utils::convert_chat_message)
            .collect();

        // Tools
        let tools = req
            .tools
            .as_ref()
            .map(|v| v.iter().map(super::utils::convert_tool).collect::<Vec<_>>());

        // Options from common params + provider params
        let options = Some(super::utils::build_model_options(
            req.common_params.temperature,
            req.common_params.max_tokens,
            req.common_params.top_p,
            None,
            None,
            self.params.options.as_ref(),
        ));

        // Structured output / format
        let mut format_val: Option<serde_json::Value> = None;
        if let Some(pp) = &req.provider_params {
            if let Some(so) = pp
                .params
                .get("structured_output")
                .and_then(|v| v.as_object())
            {
                if let Some(schema) = so.get("schema") {
                    format_val = Some(schema.clone());
                } else if let Some(t) = so.get("type").and_then(|v| v.as_str()) {
                    if t.eq_ignore_ascii_case("json") || t.eq_ignore_ascii_case("json_object") {
                        format_val = Some(serde_json::Value::String("json".to_string()));
                    }
                }
            }
        }
        if format_val.is_none() {
            if let Some(fmt) = &self.params.format {
                if fmt == "json" {
                    format_val = Some(serde_json::Value::String("json".to_string()));
                } else if let Ok(schema) = serde_json::from_str::<serde_json::Value>(fmt) {
                    format_val = Some(schema);
                } else {
                    format_val = Some(serde_json::Value::String(fmt.clone()));
                }
            }
        }

        // Thinking mode
        let think = self.params.think.or_else(|| {
            let m = req.common_params.model.to_lowercase();
            if m.contains("deepseek-r1") || m.contains("qwen3") {
                Some(true)
            } else {
                None
            }
        });

        let body = super::types::OllamaChatRequest {
            model: req.common_params.model.clone(),
            messages,
            tools,
            stream: Some(req.stream),
            format: format_val,
            options,
            keep_alive: self.params.keep_alive.clone(),
            think,
        };

        serde_json::to_value(body)
            .map_err(|e| LlmError::ParseError(format!("Serialize request failed: {e}")))
    }
}

#[derive(Clone)]
pub struct OllamaResponseTransformer;

impl ResponseTransformer for OllamaResponseTransformer {
    fn provider_id(&self) -> &str {
        "ollama"
    }

    fn transform_chat_response(&self, raw: &serde_json::Value) -> Result<ChatResponse, LlmError> {
        let response: super::types::OllamaChatResponse = serde_json::from_value(raw.clone())
            .map_err(|e| LlmError::ParseError(format!("Invalid Ollama response: {e}")))?;

        let message = super::utils::convert_from_ollama_message(&response.message);

        // Usage
        let usage = if response.prompt_eval_count.is_some() || response.eval_count.is_some() {
            Some(Usage {
                prompt_tokens: response.prompt_eval_count.unwrap_or(0),
                completion_tokens: response.eval_count.unwrap_or(0),
                total_tokens: response.prompt_eval_count.unwrap_or(0)
                    + response.eval_count.unwrap_or(0),
                cached_tokens: None,
                reasoning_tokens: None,
            })
        } else {
            None
        };

        // Finish reason
        let finish_reason = match response.done_reason.as_deref() {
            Some("stop") => Some(FinishReason::Stop),
            Some("length") => Some(FinishReason::Length),
            _ => Some(FinishReason::Stop),
        };

        // Tool calls converted in convert_from_ollama_message? It populates tool_calls field
        let tool_calls = message.tool_calls.clone();

        Ok(ChatResponse {
            id: None,
            model: Some(response.model),
            content: message.content,
            usage,
            finish_reason,
            tool_calls,
            thinking: None,
            metadata: std::collections::HashMap::new(),
        })
    }
}
