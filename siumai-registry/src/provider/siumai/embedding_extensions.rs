use super::Siumai;
use crate::core::EmbeddingCapability;
use crate::error::LlmError;
use crate::traits::EmbeddingExtensions;
use crate::types::*;

#[async_trait::async_trait]
impl EmbeddingExtensions for Siumai {
    /// Generate embeddings with advanced configuration including task types.
    ///
    /// This method enables the unified interface to support provider-specific features
    /// like Gemini's task type optimization, OpenAI's custom dimensions, etc.
    ///
    /// # Arguments
    /// * `request` - Detailed embedding request with configuration
    ///
    /// # Returns
    /// Embedding response with vectors and metadata
    ///
    /// # Example
    /// ```rust,no_run
    /// use siumai::prelude::*;
    /// use siumai::types::{EmbeddingRequest, EmbeddingTaskType};
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let client = Siumai::builder()
    ///         .gemini()
    ///         .api_key("your-api-key")
    ///         .model("gemini-embedding-001")
    ///         .build()
    ///         .await?;
    ///
    ///     // Use task type for optimization
    ///     let request = EmbeddingRequest::query("What is machine learning?");
    ///     let response = client.embed_with_config(request).await?;
    ///     Ok(())
    /// }
    /// ```
    async fn embed_with_config(
        &self,
        request: EmbeddingRequest,
    ) -> Result<EmbeddingResponse, LlmError> {
        if let Some(extensions) = self.client.as_embedding_extensions() {
            return extensions.embed_with_config(request).await;
        }

        if let Some(embedding_client) = self.client.as_embedding_capability() {
            embedding_client.embed(request.input).await
        } else {
            Err(LlmError::UnsupportedOperation(format!(
                "Provider {} does not support embedding functionality. Consider using OpenAI, Gemini, or Ollama for embeddings.",
                self.client.provider_id()
            )))
        }
    }

    async fn embed_batch(
        &self,
        requests: BatchEmbeddingRequest,
    ) -> Result<BatchEmbeddingResponse, LlmError> {
        if let Some(extensions) = self.client.as_embedding_extensions() {
            return extensions.embed_batch(requests).await;
        }

        let embedding_client = self.client.as_embedding_capability().ok_or_else(|| {
            LlmError::UnsupportedOperation(format!(
                "Provider {} does not support embedding functionality. Consider using OpenAI, Gemini, or Ollama for embeddings.",
                self.client.provider_id()
            ))
        })?;

        let mut responses = Vec::new();
        for request in requests.requests {
            let result = embedding_client
                .embed(request.input)
                .await
                .map_err(|error| error.to_string());
            responses.push(result);
            if requests.batch_options.fail_fast && responses.last().is_some_and(|r| r.is_err()) {
                break;
            }
        }

        Ok(BatchEmbeddingResponse {
            responses,
            metadata: std::collections::HashMap::new(),
        })
    }

    /// List available embedding models with their capabilities.
    ///
    /// # Returns
    /// Vector of embedding model information including supported task types
    async fn list_embedding_models(&self) -> Result<Vec<EmbeddingModelInfo>, LlmError> {
        // For now, provide basic model information based on the provider
        if let Some(_embedding_client) = self.client.as_embedding_capability() {
            let models = self.supported_embedding_models();
            let model_infos = models
                .into_iter()
                .map(|id| {
                    let mut model_info = EmbeddingModelInfo::new(
                        id.clone(),
                        id,
                        self.embedding_dimension(),
                        self.max_tokens_per_embedding(),
                    );

                    // Add task type support for Gemini models
                    if self.client.provider_id() == "gemini" {
                        model_info = model_info
                            .with_task(EmbeddingTaskType::RetrievalQuery)
                            .with_task(EmbeddingTaskType::RetrievalDocument)
                            .with_task(EmbeddingTaskType::SemanticSimilarity)
                            .with_task(EmbeddingTaskType::Classification)
                            .with_task(EmbeddingTaskType::Clustering)
                            .with_task(EmbeddingTaskType::QuestionAnswering)
                            .with_task(EmbeddingTaskType::FactVerification);
                    }

                    // Add custom dimensions support for OpenAI models
                    if self.client.provider_id() == "openai" {
                        model_info = model_info.with_custom_dimensions();
                    }

                    model_info
                })
                .collect();
            Ok(model_infos)
        } else {
            Err(LlmError::UnsupportedOperation(format!(
                "Provider {} does not support embedding functionality.",
                self.client.provider_id()
            )))
        }
    }
}
