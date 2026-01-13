//! Gemini token utilities (provider-specific)
//!
//! Implements `models/{model}:countTokens`.

use crate::error::LlmError;
use crate::execution::executors::common::{HttpBody, HttpExecutionConfig, execute_json_request};
use crate::execution::http::interceptor::HttpInterceptor;
use crate::retry_api::RetryOptions;
use crate::utils::url::join_url;
use reqwest::Client as HttpClient;
use std::sync::Arc;

use super::types::{Content, GeminiConfig, GenerateContentRequest};

fn normalize_model_id(model: &str) -> String {
    let trimmed = model.trim().trim_matches('/');
    if trimmed.is_empty() {
        return String::new();
    }
    if let Some(pos) = trimmed.rfind("/models/") {
        return trimmed[(pos + "/models/".len())..].to_string();
    }
    if let Some(rest) = trimmed.strip_prefix("models/") {
        return rest.to_string();
    }
    trimmed.to_string()
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct GeminiCountTokensRequest {
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "generateContentRequest"
    )]
    pub generate_content_request: Option<GenerateContentRequest>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub contents: Option<Vec<Content>>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct GeminiCountTokensResponse {
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "cachedContentTokenCount"
    )]
    pub cached_content_token_count: Option<i32>,
    #[serde(rename = "totalTokens")]
    pub total_tokens: i32,
    #[serde(default, rename = "promptTokensDetails")]
    pub prompt_tokens_details: Vec<serde_json::Value>,
    #[serde(default, rename = "cacheTokensDetails")]
    pub cache_tokens_details: Vec<serde_json::Value>,
}

#[derive(Clone)]
pub struct GeminiTokens {
    pub(crate) config: GeminiConfig,
    pub(crate) http_client: HttpClient,
    pub(crate) http_interceptors: Vec<std::sync::Arc<dyn HttpInterceptor>>,
    pub(crate) retry_options: Option<RetryOptions>,
}

impl GeminiTokens {
    pub fn new(
        config: GeminiConfig,
        http_client: HttpClient,
        http_interceptors: Vec<std::sync::Arc<dyn HttpInterceptor>>,
        retry_options: Option<RetryOptions>,
    ) -> Self {
        Self {
            config,
            http_client,
            http_interceptors,
            retry_options,
        }
    }

    /// Count tokens for a pre-built `GenerateContentRequest`.
    pub async fn count_tokens_for_generate_request(
        &self,
        req: GenerateContentRequest,
    ) -> Result<GeminiCountTokensResponse, LlmError> {
        let model = normalize_model_id(&req.model);
        if model.is_empty() {
            return Err(LlmError::InvalidParameter(
                "Model must be specified".to_string(),
            ));
        }

        let url = join_url(
            &self.config.base_url,
            &format!("models/{model}:countTokens"),
        );
        let body = GeminiCountTokensRequest {
            generate_content_request: Some(req),
            contents: None,
        };
        let body_json =
            serde_json::to_value(&body).map_err(|e| LlmError::JsonError(e.to_string()))?;

        let ctx = self.build_context().await;
        let http_config = self.build_http_config(ctx);
        let call = || {
            let http_config = http_config.clone();
            let url = url.clone();
            let body = body_json.clone();
            async move {
                let result =
                    execute_json_request(&http_config, &url, HttpBody::Json(body), None, false)
                        .await?;
                serde_json::from_value::<GeminiCountTokensResponse>(result.json).map_err(|e| {
                    LlmError::ParseError(format!(
                        "Failed to parse Gemini countTokens response: {e}"
                    ))
                })
            }
        };
        self.retry(call).await
    }

    async fn build_context(&self) -> crate::core::ProviderContext {
        super::context::build_context(&self.config).await
    }

    fn build_http_config(&self, ctx: crate::core::ProviderContext) -> HttpExecutionConfig {
        let mut wiring = crate::execution::wiring::HttpExecutionWiring::new(
            "gemini",
            self.http_client.clone(),
            ctx,
        )
        .with_interceptors(self.http_interceptors.clone())
        .with_retry_options(self.retry_options.clone());

        if let Some(transport) = self.config.http_transport.clone() {
            wiring = wiring.with_transport(transport);
        }

        wiring.config(Arc::new(crate::providers::gemini::spec::GeminiSpec))
    }

    async fn retry<T, F, Fut>(&self, call: F) -> Result<T, LlmError>
    where
        T: Send,
        F: Fn() -> Fut + Send + Sync,
        Fut: std::future::Future<Output = Result<T, LlmError>> + Send,
    {
        crate::retry_api::maybe_retry(self.retry_options.clone(), call).await
    }
}
