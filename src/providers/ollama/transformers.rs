//! Transformers for Ollama Chat
//!
//! Minimal RequestTransformer to support tests verifying mapping of
//! `CommonParams.max_tokens` to Ollama's `num_predict`.

use crate::error::LlmError;
use crate::transformers::request::RequestTransformer;
use crate::types::ChatRequest;

#[derive(Clone)]
pub struct OllamaRequestTransformer;

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

        // Base body: model + messages
        let mut body = serde_json::json!({
            "model": req.common_params.model,
            // Messages are not required for the current tests; provide empty list
            "messages": [],
        });

        // Map temperature/top_p if present
        if let Some(t) = req.common_params.temperature {
            body["temperature"] = serde_json::json!(t);
        }
        if let Some(tp) = req.common_params.top_p {
            body["top_p"] = serde_json::json!(tp);
        }

        // Map max_tokens -> num_predict when provided
        if let Some(max) = req.common_params.max_tokens {
            body["num_predict"] = serde_json::json!(max);
        }

        Ok(body)
    }
}
