//! Provider-specific capability traits

use crate::error::LlmError;
use crate::traits::embedding::EmbeddingCapability;
use crate::types::*;
use async_trait::async_trait;

#[async_trait]
pub trait OpenAiCapability: Send + Sync {
    async fn chat_with_structured_output(
        &self,
        messages: Vec<ChatMessage>,
        schema: JsonSchema,
    ) -> Result<StructuredResponse, LlmError>;

    async fn create_batch(&self, requests: Vec<BatchRequest>) -> Result<BatchResponse, LlmError>;

    async fn chat_with_responses_api(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<OpenAiBuiltInTool>>,
    ) -> Result<ChatResponse, LlmError>;
}

#[async_trait]
pub trait AnthropicCapability: Send + Sync {
    async fn chat_with_cache(
        &self,
        request: ChatRequest,
        cache_config: CacheConfig,
    ) -> Result<ChatResponse, LlmError>;

    async fn chat_with_thinking(&self, request: ChatRequest) -> Result<ThinkingResponse, LlmError>;
}

#[async_trait]
pub trait OpenAiEmbeddingCapability: EmbeddingCapability {
    async fn embed_with_dimensions(
        &self,
        input: Vec<String>,
        dimensions: u32,
    ) -> Result<EmbeddingResponse, LlmError>;

    async fn embed_with_format(
        &self,
        input: Vec<String>,
        format: EmbeddingFormat,
    ) -> Result<EmbeddingResponse, LlmError>;
}

#[async_trait]
pub trait GeminiCapability: Send + Sync {
    async fn chat_with_search(
        &self,
        request: ChatRequest,
        search_config: SearchConfig,
    ) -> Result<ChatResponse, LlmError>;

    async fn execute_code(
        &self,
        code: String,
        language: String,
    ) -> Result<ExecutionResponse, LlmError>;
}

#[async_trait]
pub trait GeminiEmbeddingCapability: EmbeddingCapability {
    async fn embed_with_task_type(
        &self,
        input: Vec<String>,
        task_type: EmbeddingTaskType,
    ) -> Result<EmbeddingResponse, LlmError>;

    async fn embed_with_title(
        &self,
        input: Vec<String>,
        title: String,
    ) -> Result<EmbeddingResponse, LlmError>;

    async fn embed_with_output_dimensionality(
        &self,
        input: Vec<String>,
        output_dimensionality: u32,
    ) -> Result<EmbeddingResponse, LlmError>;
}

#[async_trait]
pub trait OllamaEmbeddingCapability: EmbeddingCapability {
    async fn embed_with_model_options(
        &self,
        input: Vec<String>,
        model: String,
        options: std::collections::HashMap<String, serde_json::Value>,
    ) -> Result<EmbeddingResponse, LlmError>;

    async fn embed_with_truncation(
        &self,
        input: Vec<String>,
        truncate: bool,
    ) -> Result<EmbeddingResponse, LlmError>;

    async fn embed_with_keep_alive(
        &self,
        input: Vec<String>,
        keep_alive: String,
    ) -> Result<EmbeddingResponse, LlmError>;
}
