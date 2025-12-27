//! Transformers for Ollama (protocol layer)
//!
//! Request/Response transformers wiring to enable HttpChatExecutor path.

use crate::error::LlmError;
use crate::execution::transformers::{request::RequestTransformer, response::ResponseTransformer};
use crate::types::{ChatRequest, ChatResponse, EmbeddingRequest, EmbeddingResponse};

#[derive(Clone)]
pub struct OllamaRequestTransformer {
    pub params: super::params::OllamaParams,
}

impl RequestTransformer for OllamaRequestTransformer {
    fn provider_id(&self) -> &str {
        "ollama"
    }

    fn transform_chat(&self, req: &ChatRequest) -> Result<serde_json::Value, LlmError> {
        let body = super::utils::build_chat_request(req, &self.params)?;
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
        Ok(super::utils::convert_chat_response(response))
    }
}

#[derive(Clone)]
pub struct OllamaEmbeddingRequestTransformer {
    pub default_model: String,
    pub params: super::params::OllamaParams,
}

impl RequestTransformer for OllamaEmbeddingRequestTransformer {
    fn provider_id(&self) -> &str {
        "ollama"
    }

    fn transform_chat(&self, _req: &ChatRequest) -> Result<serde_json::Value, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "ollama embedding transformer does not implement chat".into(),
        ))
    }

    fn transform_embedding(&self, req: &EmbeddingRequest) -> Result<serde_json::Value, LlmError> {
        let body = super::utils::build_embedding_request(req, &self.default_model, &self.params)?;
        serde_json::to_value(body)
            .map_err(|e| LlmError::ParseError(format!("Serialize request failed: {e}")))
    }
}

#[derive(Clone)]
pub struct OllamaEmbeddingResponseTransformer;

impl ResponseTransformer for OllamaEmbeddingResponseTransformer {
    fn provider_id(&self) -> &str {
        "ollama"
    }

    fn transform_embedding_response(
        &self,
        raw: &serde_json::Value,
    ) -> Result<EmbeddingResponse, LlmError> {
        let response: super::types::OllamaEmbeddingResponse =
            serde_json::from_value(raw.clone())
                .map_err(|e| LlmError::ParseError(format!("Invalid Ollama response: {e}")))?;
        Ok(super::utils::convert_embedding_response(response))
    }

    fn transform_chat_response(&self, _raw: &serde_json::Value) -> Result<ChatResponse, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "ollama embedding transformer does not implement chat response".into(),
        ))
    }
}
