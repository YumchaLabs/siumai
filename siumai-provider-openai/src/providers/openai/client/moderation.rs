//! OpenAI moderation capability implementation (delegating to `OpenAiModeration`).

use async_trait::async_trait;

use crate::error::LlmError;
use crate::traits::ModerationCapability;
use crate::types::{ModerationRequest, ModerationResponse};

use super::OpenAiClient;

impl OpenAiClient {
    /// Provider-specific: OpenAI multi-modal moderation (`input` array of objects).
    ///
    /// This is intentionally not part of the unified `ModerationRequest`, because the
    /// multi-modal input shape is provider-specific and doesn't translate well.
    pub(crate) async fn moderate_multimodal(
        &self,
        request: crate::providers::openai::ext::moderation::OpenAiModerationRequest,
    ) -> Result<ModerationResponse, LlmError> {
        use crate::execution::transformers::response::ResponseTransformer;
        use crate::execution::executors::common::{execute_json_request, HttpBody, HttpExecutionConfig};
        use secrecy::ExposeSecret;
        use std::sync::Arc;

        if request.input.is_empty() {
            return Err(LlmError::InvalidInput(
                "OpenAI multimodal moderation input cannot be empty".to_string(),
            ));
        }

        // Minimal stable validation: reject empty text/url entries.
        for item in &request.input {
            match item {
                crate::providers::openai::ext::moderation::OpenAiModerationInput::Text { text } => {
                    if text.trim().is_empty() {
                        return Err(LlmError::InvalidInput(
                            "OpenAI moderation text input cannot be empty".to_string(),
                        ));
                    }
                }
                crate::providers::openai::ext::moderation::OpenAiModerationInput::ImageUrl {
                    image_url,
                } => {
                    if image_url.url.trim().is_empty() {
                        return Err(LlmError::InvalidInput(
                            "OpenAI moderation image_url cannot be empty".to_string(),
                        ));
                    }
                }
            }
        }

        let model = request
            .model
            .clone()
            .unwrap_or_else(|| "omni-moderation-latest".to_string());
        let input = serde_json::to_value(&request.input)
            .map_err(|e| LlmError::InvalidParameter(format!("Invalid moderation input: {e}")))?;

        let spec = Arc::new(crate::providers::openai::spec::OpenAiSpec::new());
        let ctx = crate::core::ProviderContext::new(
            "openai",
            self.base_url.clone(),
            Some(self.api_key.expose_secret().to_string()),
            self.http_config.headers.clone(),
        )
        .with_org_project(self.organization.clone(), self.project.clone());

        let config = HttpExecutionConfig {
            provider_id: "openai".to_string(),
            http_client: self.http_client.clone(),
            provider_spec: spec,
            provider_context: ctx,
            interceptors: self.http_interceptors.clone(),
            retry_options: self.retry_options.clone(),
        };

        let url = format!("{}/moderations", self.base_url.trim_end_matches('/'));
        let body = serde_json::json!({ "model": model, "input": input });

        let result = execute_json_request(&config, &url, HttpBody::Json(body), None, false).await?;

        let resp_tx = crate::providers::openai::transformers::OpenAiResponseTransformer;
        resp_tx.transform_moderation_response(&result.json)
    }
}

#[async_trait]
impl ModerationCapability for OpenAiClient {
    async fn moderate(&self, request: ModerationRequest) -> Result<ModerationResponse, LlmError> {
        let cfg = super::super::OpenAiConfig {
            api_key: self.api_key.clone(),
            base_url: self.base_url.clone(),
            organization: self.organization.clone(),
            project: self.project.clone(),
            common_params: self.common_params.clone(),
            openai_params: self.openai_params.clone(),
            provider_options_map: self.default_provider_options_map.clone(),
            http_config: self.http_config.clone(),
        };

        let moderation =
            super::super::moderation::OpenAiModeration::new(cfg, self.http_client.clone());
        moderation.moderate(request).await
    }
}
