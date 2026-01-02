//! Ollama Models Capability Implementation
//!
//! Implements model management functionality for Ollama.

use async_trait::async_trait;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::error::LlmError;
use crate::traits::ModelListingCapability;
use crate::types::*;

use super::types::*;
use super::utils::*;
use crate::execution::executors::common::{HttpBody, execute_get_request, execute_json_request};

/// Ollama Models Capability Implementation
#[derive(Clone)]
pub struct OllamaModelsCapability {
    pub base_url: String,
    pub http_client: reqwest::Client,
    pub http_config: HttpConfig,
}

impl OllamaModelsCapability {
    /// Creates a new Ollama models capability
    pub const fn new(
        base_url: String,
        http_client: reqwest::Client,
        http_config: HttpConfig,
    ) -> Self {
        Self {
            base_url,
            http_client,
            http_config,
        }
    }

    /// Convert Ollama model to common `ModelInfo`
    fn convert_model_info(&self, model: &OllamaModel) -> ModelInfo {
        ModelInfo {
            id: model.name.clone(),
            name: Some(model.name.clone()),
            description: Some(format!("Ollama model: {}", model.details.family)),
            capabilities: vec!["chat".to_string(), "completion".to_string()],
            context_window: None,    // Not provided by Ollama API
            max_output_tokens: None, // Not provided by Ollama API
            created: Some(0),        // Ollama doesn't provide creation timestamp
            owned_by: "ollama".to_string(),
            input_cost_per_token: None,  // Ollama is free/local
            output_cost_per_token: None, // Ollama is free/local
        }
    }

    fn build_context(&self) -> crate::core::ProviderContext {
        crate::core::ProviderContext::new(
            "ollama",
            self.base_url.clone(),
            None,
            self.http_config.headers.clone(),
        )
    }

    fn build_http_config(
        &self,
        ctx: crate::core::ProviderContext,
    ) -> crate::execution::executors::common::HttpExecutionConfig {
        let spec = Arc::new(crate::providers::ollama::spec::OllamaSpec::new(
            crate::providers::ollama::config::OllamaParams::default(),
        ));
        crate::execution::wiring::HttpExecutionWiring::new("ollama", self.http_client.clone(), ctx)
            .config(spec)
    }

    /// Pull a model from Ollama registry
    pub async fn pull_model(&self, model_name: String) -> Result<(), LlmError> {
        validate_model_name(&model_name)?;

        let url = format!("{}/api/pull", self.base_url.trim_end_matches('/'));
        let body = serde_json::json!({ "model": model_name, "stream": false });

        let ctx = self.build_context();
        let config = self.build_http_config(ctx);
        execute_json_request(&config, &url, HttpBody::Json(body), None, false).await?;
        Ok(())
    }

    /// Pull a model with streaming progress
    pub async fn pull_model_stream(
        &self,
        model_name: String,
    ) -> Result<impl futures_util::Stream<Item = Result<PullProgress, LlmError>>, LlmError> {
        validate_model_name(&model_name)?;

        let url = format!("{}/api/pull", self.base_url.trim_end_matches('/'));
        let body = serde_json::json!({ "model": model_name, "stream": true });

        let ctx = self.build_context();
        let config = self.build_http_config(ctx);
        let response =
            crate::execution::executors::common::execute_json_request_streaming_response(
                &config, &url, body, None,
            )
            .await?;

        let stream = response
            .bytes_stream()
            .map(|chunk_result| match chunk_result {
                Ok(chunk) => {
                    let chunk_str = String::from_utf8_lossy(&chunk);
                    for line in chunk_str.lines() {
                        if let Ok(Some(json_value)) = parse_streaming_line(line)
                            && let Ok(progress) = serde_json::from_value::<PullProgress>(json_value)
                        {
                            return Ok(progress);
                        }
                    }
                    Ok(PullProgress {
                        status: "processing".to_string(),
                        digest: None,
                        total: None,
                        completed: None,
                    })
                }
                Err(e) => Err(LlmError::StreamError(format!("Stream error: {e}"))),
            });

        Ok(stream)
    }

    /// Delete a model
    pub async fn delete_model(&self, model_name: String) -> Result<(), LlmError> {
        validate_model_name(&model_name)?;

        let url = format!("{}/api/delete", self.base_url.trim_end_matches('/'));
        let body = serde_json::json!({ "model": model_name });

        let ctx = self.build_context();
        let config = self.build_http_config(ctx);
        crate::execution::executors::common::execute_delete_json_request(&config, &url, body, None)
            .await?;
        Ok(())
    }

    /// Copy a model
    pub async fn copy_model(&self, source: String, destination: String) -> Result<(), LlmError> {
        validate_model_name(&source)?;
        validate_model_name(&destination)?;

        let url = format!("{}/api/copy", self.base_url.trim_end_matches('/'));
        let body = serde_json::json!({ "source": source, "destination": destination });

        let ctx = self.build_context();
        let config = self.build_http_config(ctx);
        execute_json_request(&config, &url, HttpBody::Json(body), None, false).await?;
        Ok(())
    }

    /// Show model information
    pub async fn show_model(&self, model_name: String) -> Result<ModelDetails, LlmError> {
        validate_model_name(&model_name)?;

        let url = crate::utils::url::join_url(&self.base_url, "api/show");

        let body = serde_json::json!({
            "model": model_name
        });

        let ctx = self.build_context();
        let config = self.build_http_config(ctx);
        let result = execute_json_request(&config, &url, HttpBody::Json(body), None, false).await?;
        let model_info = result.json;

        Ok(ModelDetails {
            modelfile: model_info
                .get("modelfile")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            parameters: model_info
                .get("parameters")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            template: model_info
                .get("template")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            details: model_info
                .get("details")
                .cloned()
                .unwrap_or(serde_json::Value::Null),
        })
    }

    /// List running models
    pub async fn list_running_models(&self) -> Result<Vec<RunningModelInfo>, LlmError> {
        let url = crate::utils::url::join_url(&self.base_url, "api/ps");
        let ctx = self.build_context();
        let config = self.build_http_config(ctx);
        let result = execute_get_request(&config, &url, None).await?;
        let running_response: OllamaRunningModelsResponse = serde_json::from_value(result.json)
            .map_err(|e| {
                LlmError::ParseError(format!(
                    "Failed to parse Ollama running models response: {e}"
                ))
            })?;

        let running_models = running_response
            .models
            .into_iter()
            .map(|model| RunningModelInfo {
                name: model.name,
                model: model.model,
                size: model.size,
                digest: model.digest,
                expires_at: model.expires_at,
                size_vram: model.size_vram,
            })
            .collect();

        Ok(running_models)
    }
}

#[async_trait]
impl ModelListingCapability for OllamaModelsCapability {
    async fn list_models(&self) -> Result<Vec<ModelInfo>, LlmError> {
        let ctx = self.build_context();
        let config = self.build_http_config(ctx);
        let url = config.provider_spec.models_url(&config.provider_context);
        let result = execute_get_request(&config, &url, None).await?;
        let models_response: OllamaModelsResponse =
            serde_json::from_value(result.json).map_err(|e| {
                LlmError::ParseError(format!("Failed to parse Ollama models response: {e}"))
            })?;

        let models = models_response
            .models
            .into_iter()
            .map(|model| self.convert_model_info(&model))
            .collect();

        Ok(models)
    }

    async fn get_model(&self, model_id: String) -> Result<ModelInfo, LlmError> {
        // First try to find the model in the list
        let models = self.list_models().await?;

        for model in models {
            if model.id == model_id {
                return Ok(model);
            }
        }

        // If not found, return NotFound error
        Err(LlmError::NotFound(format!("Model '{model_id}' not found")))
    }
}

/// Pull progress information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PullProgress {
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub digest: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completed: Option<u64>,
}

/// Model details information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDetails {
    pub modelfile: String,
    pub parameters: String,
    pub template: String,
    pub details: serde_json::Value,
}

/// Running model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunningModelInfo {
    pub name: String,
    pub model: String,
    pub size: u64,
    pub digest: String,
    pub expires_at: String,
    pub size_vram: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convert_model_info() {
        let capability = OllamaModelsCapability::new(
            "http://localhost:11434".to_string(),
            reqwest::Client::new(),
            HttpConfig::default(),
        );

        let ollama_model = OllamaModel {
            name: "llama3.2:latest".to_string(),
            model: "llama3.2:latest".to_string(),
            modified_at: "2023-01-01T00:00:00Z".to_string(),
            size: 1_000_000,
            digest: "sha256:abc123".to_string(),
            details: OllamaModelDetails {
                parent_model: "".to_string(),
                format: "gguf".to_string(),
                family: "llama".to_string(),
                families: vec!["llama".to_string()],
                parameter_size: "3.2B".to_string(),
                quantization_level: "Q4_K_M".to_string(),
            },
        };

        let model_info = capability.convert_model_info(&ollama_model);
        assert_eq!(model_info.id, "llama3.2:latest");
        assert_eq!(model_info.owned_by, "ollama");
        assert_eq!(model_info.name, Some("llama3.2:latest".to_string()));
    }
}
