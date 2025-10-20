//! Ollama Completion Capability Implementation
//!
//! Implements text completion using the /api/generate endpoint.

use crate::error::LlmError;
use crate::stream::ChatStream;
use crate::types::*;

use super::config::OllamaParams;
use super::streaming::OllamaStreaming;
use super::types::*;
use super::utils::*;

/// Ollama Completion Capability Implementation
#[derive(Clone)]
pub struct OllamaCompletionCapability {
    pub base_url: String,
    pub http_client: reqwest::Client,
    pub http_config: HttpConfig,
    pub ollama_params: OllamaParams,
    streaming: OllamaStreaming,
}

impl OllamaCompletionCapability {
    /// Creates a new Ollama completion capability
    pub fn new(
        base_url: String,
        http_client: reqwest::Client,
        http_config: HttpConfig,
        ollama_params: OllamaParams,
    ) -> Self {
        let streaming = OllamaStreaming::new(http_client.clone());
        Self {
            base_url,
            http_client,
            http_config,
            ollama_params,
            streaming,
        }
    }

    /// Build generate request body
    fn build_generate_request_body(
        &self,
        prompt: &str,
        model: Option<&str>,
        stream: bool,
    ) -> Result<OllamaGenerateRequest, LlmError> {
        let model = model
            .or({
                // Try to get model from ollama_params or use default
                None
            })
            .unwrap_or("llama3.2")
            .to_string();

        validate_model_name(&model)?;

        // Build model options
        let options = build_model_options(
            None, // temperature
            None, // max_tokens
            None, // top_p
            None, // frequency_penalty
            None, // presence_penalty
            self.ollama_params.options.as_ref(),
        );

        // Build format if specified
        let format = if let Some(format_str) = &self.ollama_params.format {
            if format_str == "json" {
                Some(serde_json::Value::String("json".to_string()))
            } else {
                // Try to parse as JSON schema
                match serde_json::from_str(format_str) {
                    Ok(schema) => Some(schema),
                    Err(_) => Some(serde_json::Value::String(format_str.clone())),
                }
            }
        } else {
            None
        };

        Ok(OllamaGenerateRequest {
            model,
            prompt: prompt.to_string(),
            suffix: None,
            images: None,
            stream: Some(stream),
            format,
            options: if options.is_empty() {
                None
            } else {
                Some(options)
            },
            system: None,
            template: None,
            raw: self.ollama_params.raw,
            keep_alive: self.ollama_params.keep_alive.clone(),
            context: None,
            think: self.ollama_params.think,
        })
    }

    /// Parse generate response
    fn parse_generate_response(&self, response: OllamaGenerateResponse) -> String {
        response.response
    }

    /// Generate text completion
    pub async fn generate(&self, prompt: String) -> Result<String, LlmError> {
        let headers = build_headers(&self.http_config.headers)?;
        let body = self.build_generate_request_body(&prompt, None, false)?;
        let url = format!("{}/api/generate", self.base_url);

        let response = self
            .http_client
            .post(&url)
            .headers(headers)
            .json(&body)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(LlmError::HttpError(format!(
                "Generate request failed: {status} - {error_text}"
            )));
        }

        let ollama_response: OllamaGenerateResponse = response.json().await?;
        Ok(self.parse_generate_response(ollama_response))
    }

    /// Generate text completion with streaming
    pub async fn generate_stream(&self, prompt: String) -> Result<ChatStream, LlmError> {
        let headers = build_headers(&self.http_config.headers)?;
        let body = self.build_generate_request_body(&prompt, None, true)?;
        let url = format!("{}/api/generate", self.base_url);

        // Use the dedicated streaming capability
        self.streaming
            .clone()
            .create_completion_stream(url, headers, body)
            .await
    }

    /// Generate with custom model
    pub async fn generate_with_model(
        &self,
        prompt: String,
        model: String,
    ) -> Result<String, LlmError> {
        let headers = build_headers(&self.http_config.headers)?;
        let body = self.build_generate_request_body(&prompt, Some(&model), false)?;
        let url = crate::utils::url::join_url(&self.base_url, "api/generate");

        let response = self
            .http_client
            .post(&url)
            .headers(headers)
            .json(&body)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(LlmError::HttpError(format!(
                "Generate request failed: {status} - {error_text}"
            )));
        }

        let ollama_response: OllamaGenerateResponse = response.json().await?;
        Ok(self.parse_generate_response(ollama_response))
    }

    /// Generate with suffix (for code completion)
    pub async fn generate_with_suffix(
        &self,
        prompt: String,
        suffix: String,
        model: Option<String>,
    ) -> Result<String, LlmError> {
        let headers = build_headers(&self.http_config.headers)?;
        let mut body = self.build_generate_request_body(&prompt, model.as_deref(), false)?;
        body.suffix = Some(suffix);
        let url = format!("{}/api/generate", self.base_url);

        let response = self
            .http_client
            .post(&url)
            .headers(headers)
            .json(&body)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(LlmError::HttpError(format!(
                "Generate request failed: {status} - {error_text}"
            )));
        }

        let ollama_response: OllamaGenerateResponse = response.json().await?;
        Ok(self.parse_generate_response(ollama_response))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_generate_request_body() {
        let capability = OllamaCompletionCapability::new(
            "http://localhost:11434".to_string(),
            reqwest::Client::new(),
            HttpConfig::default(),
            OllamaParams::default(),
        );

        let body = capability
            .build_generate_request_body("Hello world", Some("llama3.2"), false)
            .unwrap();
        assert_eq!(body.model, "llama3.2");
        assert_eq!(body.prompt, "Hello world");
        assert_eq!(body.stream, Some(false));
    }

    #[test]
    fn test_parse_generate_response() {
        let capability = OllamaCompletionCapability::new(
            "http://localhost:11434".to_string(),
            reqwest::Client::new(),
            HttpConfig::default(),
            OllamaParams::default(),
        );

        let ollama_response = OllamaGenerateResponse {
            model: "llama3.2".to_string(),
            created_at: "2023-01-01T00:00:00Z".to_string(),
            response: "Hello there!".to_string(),
            done: true,
            context: None,
            total_duration: Some(1_000_000_000),
            load_duration: Some(100_000_000),
            prompt_eval_count: Some(10),
            prompt_eval_duration: Some(200_000_000),
            eval_count: Some(20),
            eval_duration: Some(700_000_000),
        };

        let response = capability.parse_generate_response(ollama_response);
        assert_eq!(response, "Hello there!");
    }
}
