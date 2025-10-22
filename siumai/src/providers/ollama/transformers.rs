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
        let tools = req.tools.as_ref().map(|v| {
            v.iter()
                .filter_map(super::utils::convert_tool)
                .collect::<Vec<_>>()
        });

        // Options from common params + provider params
        let options = Some(super::utils::build_model_options(
            req.common_params.temperature,
            req.common_params.max_tokens,
            req.common_params.top_p,
            None,
            None,
            self.params.options.as_ref(),
        ));

        // Structured output / format - now handled via provider_options
        let format_val: Option<serde_json::Value> = if let Some(fmt) = &self.params.format {
            if fmt == "json" {
                Some(serde_json::Value::String("json".to_string()))
            } else if let Ok(schema) = serde_json::from_str::<serde_json::Value>(fmt) {
                Some(schema)
            } else {
                Some(serde_json::Value::String(fmt.clone()))
            }
        } else {
            None
        };

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
            let prompt = response.prompt_eval_count.unwrap_or(0);
            let completion = response.eval_count.unwrap_or(0);
            Some(
                Usage::builder()
                    .prompt_tokens(prompt)
                    .completion_tokens(completion)
                    .total_tokens(prompt + completion)
                    .build(),
            )
        } else {
            None
        };

        // Finish reason
        let finish_reason = match response.done_reason.as_deref() {
            Some("stop") => Some(FinishReason::Stop),
            Some("length") => Some(FinishReason::Length),
            _ => Some(FinishReason::Stop),
        };

        // Tool calls and thinking are now part of message.content (converted in convert_from_ollama_message)
        Ok(ChatResponse {
            id: None,
            model: Some(response.model),
            content: message.content,
            usage,
            finish_reason,
            audio: None, // Ollama doesn't support audio output
            system_fingerprint: None,
            service_tier: None,
            warnings: None,
            metadata: std::collections::HashMap::new(),
        })
    }
}
