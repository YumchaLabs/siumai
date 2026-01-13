//! Anthropic Message Batches API (provider-specific).
//!
//! Implements:
//! - `POST /v1/messages/batches`
//! - `GET /v1/messages/batches/{message_batch_id}`
//! - `GET /v1/messages/batches`
//! - `POST /v1/messages/batches/{message_batch_id}/cancel`
//! - `DELETE /v1/messages/batches/{message_batch_id}`
//! - `GET /v1/messages/batches/{message_batch_id}/results` (JSONL; returned as bytes)

use crate::error::LlmError;
use crate::execution::executors::common::{
    HttpBinaryResult, HttpBody, HttpExecutionConfig, execute_delete_request, execute_get_binary,
    execute_get_request, execute_json_request,
};
use crate::execution::http::interceptor::HttpInterceptor;
use crate::retry_api::RetryOptions;
use crate::utils::url::join_url;
use reqwest::Client as HttpClient;
use secrecy::{ExposeSecret, SecretString};
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Debug, Clone, serde::Serialize)]
pub struct AnthropicMessageBatchRequest {
    pub custom_id: String,
    pub params: serde_json::Value,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct AnthropicCreateMessageBatchRequest {
    pub requests: Vec<AnthropicMessageBatchRequest>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct AnthropicMessageBatch {
    pub id: Option<String>,
    #[serde(rename = "type")]
    pub r#type: Option<String>,
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct AnthropicListMessageBatchesResponse {
    #[serde(default)]
    pub data: Vec<AnthropicMessageBatch>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub first_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_id: Option<String>,
    #[serde(default)]
    pub has_more: bool,
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

#[derive(Clone)]
pub struct AnthropicMessageBatches {
    pub(crate) api_key: SecretString,
    pub(crate) base_url: String,
    pub(crate) http_client: HttpClient,
    pub(crate) http_config: crate::types::HttpConfig,
    pub(crate) beta_features: Vec<String>,
    pub(crate) http_transport: Option<Arc<dyn crate::execution::http::transport::HttpTransport>>,
    pub(crate) http_interceptors: Vec<Arc<dyn HttpInterceptor>>,
    pub(crate) retry_options: Option<RetryOptions>,
}

impl AnthropicMessageBatches {
    pub fn new(
        api_key: SecretString,
        base_url: String,
        http_client: HttpClient,
        http_config: crate::types::HttpConfig,
        beta_features: Vec<String>,
        http_transport: Option<Arc<dyn crate::execution::http::transport::HttpTransport>>,
        http_interceptors: Vec<Arc<dyn HttpInterceptor>>,
        retry_options: Option<RetryOptions>,
    ) -> Self {
        Self {
            api_key,
            base_url,
            http_client,
            http_config,
            beta_features,
            http_transport,
            http_interceptors,
            retry_options,
        }
    }

    pub async fn create(
        &self,
        req: AnthropicCreateMessageBatchRequest,
        per_request_headers: Option<HashMap<String, String>>,
    ) -> Result<AnthropicMessageBatch, LlmError> {
        let body = serde_json::to_value(req).map_err(|e| LlmError::JsonError(e.to_string()))?;
        self.create_raw(body, per_request_headers).await
    }

    pub async fn create_raw(
        &self,
        body: serde_json::Value,
        per_request_headers: Option<HashMap<String, String>>,
    ) -> Result<AnthropicMessageBatch, LlmError> {
        let url = join_url(&self.base_url, "messages/batches");
        let ctx = self.build_context();
        let http_config = self.build_http_config(ctx);

        let call = || {
            let http_config = http_config.clone();
            let url = url.clone();
            let body = body.clone();
            let headers = per_request_headers.clone();
            async move {
                let result = execute_json_request(
                    &http_config,
                    &url,
                    HttpBody::Json(body),
                    headers.as_ref(),
                    false,
                )
                .await?;
                serde_json::from_value::<AnthropicMessageBatch>(result.json).map_err(|e| {
                    LlmError::ParseError(format!(
                        "Failed to parse Anthropic message batch create response: {e}"
                    ))
                })
            }
        };

        crate::retry_api::maybe_retry(self.retry_options.clone(), call).await
    }

    pub async fn get(
        &self,
        message_batch_id: String,
        per_request_headers: Option<HashMap<String, String>>,
    ) -> Result<AnthropicMessageBatch, LlmError> {
        let url = join_url(
            &self.base_url,
            &format!("messages/batches/{message_batch_id}"),
        );
        let ctx = self.build_context();
        let http_config = self.build_http_config(ctx);

        let call = || {
            let http_config = http_config.clone();
            let url = url.clone();
            let headers = per_request_headers.clone();
            async move {
                let result = execute_get_request(&http_config, &url, headers.as_ref()).await?;
                serde_json::from_value::<AnthropicMessageBatch>(result.json).map_err(|e| {
                    LlmError::ParseError(format!(
                        "Failed to parse Anthropic message batch get response: {e}"
                    ))
                })
            }
        };

        crate::retry_api::maybe_retry(self.retry_options.clone(), call).await
    }

    pub async fn list(
        &self,
        before_id: Option<String>,
        after_id: Option<String>,
        limit: Option<u32>,
        per_request_headers: Option<HashMap<String, String>>,
    ) -> Result<AnthropicListMessageBatchesResponse, LlmError> {
        let mut url = join_url(&self.base_url, "messages/batches");
        let mut qs = Vec::new();
        if let Some(before) = before_id {
            qs.push(format!("before_id={}", urlencoding::encode(&before)));
        }
        if let Some(after) = after_id {
            qs.push(format!("after_id={}", urlencoding::encode(&after)));
        }
        if let Some(limit) = limit {
            qs.push(format!("limit={limit}"));
        }
        if !qs.is_empty() {
            url.push('?');
            url.push_str(&qs.join("&"));
        }

        let ctx = self.build_context();
        let http_config = self.build_http_config(ctx);

        let call = || {
            let http_config = http_config.clone();
            let url = url.clone();
            let headers = per_request_headers.clone();
            async move {
                let result = execute_get_request(&http_config, &url, headers.as_ref()).await?;
                serde_json::from_value::<AnthropicListMessageBatchesResponse>(result.json).map_err(
                    |e| {
                        LlmError::ParseError(format!(
                            "Failed to parse Anthropic message batches list response: {e}"
                        ))
                    },
                )
            }
        };

        crate::retry_api::maybe_retry(self.retry_options.clone(), call).await
    }

    pub async fn cancel(
        &self,
        message_batch_id: String,
        per_request_headers: Option<HashMap<String, String>>,
    ) -> Result<AnthropicMessageBatch, LlmError> {
        let url = join_url(
            &self.base_url,
            &format!("messages/batches/{message_batch_id}/cancel"),
        );
        let ctx = self.build_context();
        let http_config = self.build_http_config(ctx);

        let call = || {
            let http_config = http_config.clone();
            let url = url.clone();
            let headers = per_request_headers.clone();
            async move {
                let result = execute_json_request(
                    &http_config,
                    &url,
                    HttpBody::Json(serde_json::json!({})),
                    headers.as_ref(),
                    false,
                )
                .await?;
                serde_json::from_value::<AnthropicMessageBatch>(result.json).map_err(|e| {
                    LlmError::ParseError(format!(
                        "Failed to parse Anthropic message batch cancel response: {e}"
                    ))
                })
            }
        };

        crate::retry_api::maybe_retry(self.retry_options.clone(), call).await
    }

    pub async fn delete(
        &self,
        message_batch_id: String,
        per_request_headers: Option<HashMap<String, String>>,
    ) -> Result<(), LlmError> {
        let url = join_url(
            &self.base_url,
            &format!("messages/batches/{message_batch_id}"),
        );
        let ctx = self.build_context();
        let http_config = self.build_http_config(ctx);

        let call = || {
            let http_config = http_config.clone();
            let url = url.clone();
            let headers = per_request_headers.clone();
            async move {
                let _ = execute_delete_request(&http_config, &url, headers.as_ref()).await?;
                Ok(())
            }
        };

        crate::retry_api::maybe_retry(self.retry_options.clone(), call).await
    }

    /// Download batch results as bytes.
    ///
    /// The official API returns JSONL; this helper returns raw bytes so callers can stream/parse.
    pub async fn get_results(
        &self,
        message_batch_id: String,
        per_request_headers: Option<HashMap<String, String>>,
    ) -> Result<HttpBinaryResult, LlmError> {
        let url = join_url(
            &self.base_url,
            &format!("messages/batches/{message_batch_id}/results"),
        );
        let ctx = self.build_context();
        let http_config = self.build_http_config(ctx);

        let call = || {
            let http_config = http_config.clone();
            let url = url.clone();
            let headers = per_request_headers.clone();
            async move { execute_get_binary(&http_config, &url, headers.as_ref()).await }
        };

        crate::retry_api::maybe_retry(self.retry_options.clone(), call).await
    }

    fn build_context(&self) -> crate::core::ProviderContext {
        let mut headers = self.http_config.headers.clone();

        if !self.beta_features.is_empty() {
            let mut existing_values: Vec<String> = Vec::new();
            let keys: Vec<String> = headers
                .keys()
                .filter(|k| k.eq_ignore_ascii_case("anthropic-beta"))
                .cloned()
                .collect();
            for k in keys {
                if let Some(v) = headers.remove(&k) {
                    existing_values.push(v);
                }
            }

            let mut merged = String::new();
            if !existing_values.is_empty() {
                merged.push_str(&existing_values.join(","));
            }
            if !merged.is_empty() {
                merged.push(',');
            }
            merged.push_str(&self.beta_features.join(","));

            headers.insert("anthropic-beta".to_string(), merged);
        }

        crate::core::ProviderContext::new(
            "anthropic",
            self.base_url.clone(),
            Some(self.api_key.expose_secret().to_string()),
            headers,
        )
    }

    fn build_http_config(&self, ctx: crate::core::ProviderContext) -> HttpExecutionConfig {
        let mut wiring = crate::execution::wiring::HttpExecutionWiring::new(
            "anthropic",
            self.http_client.clone(),
            ctx,
        )
        .with_interceptors(self.http_interceptors.clone())
        .with_retry_options(self.retry_options.clone());

        if let Some(transport) = self.http_transport.clone() {
            wiring = wiring.with_transport(transport);
        }

        wiring.config(Arc::new(super::spec::AnthropicSpec::new()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_request_serializes_requests_array() {
        let req = AnthropicCreateMessageBatchRequest {
            requests: vec![AnthropicMessageBatchRequest {
                custom_id: "id-1".to_string(),
                params: serde_json::json!({"model":"claude-3-5-sonnet-20241022","max_tokens": 1, "messages":[]}),
            }],
        };

        let v = serde_json::to_value(req).expect("json");
        assert!(v.get("requests").and_then(|v| v.as_array()).is_some());
        let first = &v["requests"][0];
        assert_eq!(
            first.get("custom_id").and_then(|v| v.as_str()),
            Some("id-1")
        );
        assert!(first.get("params").is_some());
    }

    #[test]
    fn build_context_merges_beta_features_into_anthropic_beta_header() {
        let mut cfg = crate::types::HttpConfig::default();
        cfg.headers
            .insert("anthropic-beta".to_string(), "x".to_string());

        let batches = AnthropicMessageBatches::new(
            SecretString::from("k".to_string()),
            "https://api.anthropic.com/v1".to_string(),
            reqwest::Client::new(),
            cfg,
            vec!["y".to_string()],
            None,
            vec![],
            None,
        );

        let ctx = batches.build_context();
        assert_eq!(
            ctx.http_extra_headers
                .get("anthropic-beta")
                .map(|s| s.as_str()),
            Some("x,y")
        );
    }
}
