//! OpenAI Responses API management endpoints (provider-specific).
//!
//! These endpoints are intentionally not part of the Vercel-aligned unified surface.
//! They operate on the raw Responses resources (retrieve/cancel/delete/input_items/compact).

use super::OpenAiClient;
use crate::error::LlmError;
use crate::execution::executors::common::{
    HttpBody, HttpExecutionConfig, execute_delete_request, execute_get_request,
    execute_json_request,
};
use std::sync::Arc;

impl OpenAiClient {
    fn responses_admin_config(&self) -> HttpExecutionConfig {
        let spec: Arc<dyn crate::core::ProviderSpec> =
            Arc::new(crate::providers::openai::spec::OpenAiSpec::new());
        self.http_wiring().config(spec)
    }

    fn responses_url(&self, suffix: &str) -> String {
        format!(
            "{}/{}",
            self.base_url.trim_end_matches('/'),
            suffix.trim_start_matches('/')
        )
    }

    fn with_query_params(
        &self,
        url: &str,
        params: Vec<(String, String)>,
    ) -> Result<String, LlmError> {
        let mut u = reqwest::Url::parse(url).map_err(|e| {
            LlmError::InvalidParameter(format!("Invalid OpenAI base URL '{url}': {e}"))
        })?;
        {
            let mut qp = u.query_pairs_mut();
            for (k, v) in params {
                qp.append_pair(&k, &v);
            }
        }
        Ok(u.to_string())
    }

    pub(crate) async fn responses_retrieve(
        &self,
        response_id: &str,
        include: Option<Vec<String>>,
    ) -> Result<serde_json::Value, LlmError> {
        let config = self.responses_admin_config();
        let url = self.responses_url(&format!("responses/{response_id}"));

        let mut params: Vec<(String, String)> = Vec::new();
        if let Some(items) = include {
            for item in items {
                params.push(("include".to_string(), item));
            }
        }
        let url = if params.is_empty() {
            url
        } else {
            self.with_query_params(&url, params)?
        };

        let res = execute_get_request(&config, &url, None).await?;
        Ok(res.json)
    }

    pub(crate) async fn responses_delete(
        &self,
        response_id: &str,
    ) -> Result<serde_json::Value, LlmError> {
        let config = self.responses_admin_config();
        let url = self.responses_url(&format!("responses/{response_id}"));
        let res = execute_delete_request(&config, &url, None).await?;
        Ok(res.json)
    }

    pub(crate) async fn responses_cancel(
        &self,
        response_id: &str,
    ) -> Result<serde_json::Value, LlmError> {
        let config = self.responses_admin_config();
        let url = self.responses_url(&format!("responses/{response_id}/cancel"));
        let res = execute_json_request(
            &config,
            &url,
            HttpBody::Json(serde_json::json!({})),
            None,
            false,
        )
        .await?;
        Ok(res.json)
    }

    pub(crate) async fn responses_list_input_items(
        &self,
        response_id: &str,
        params: crate::providers::openai::ext::responses::OpenAiResponsesInputItemsParams,
    ) -> Result<crate::providers::openai::ext::responses::OpenAiResponsesInputItemsPage, LlmError>
    {
        let config = self.responses_admin_config();
        let url = self.responses_url(&format!("responses/{response_id}/input_items"));

        let mut qp: Vec<(String, String)> = Vec::new();
        if let Some(limit) = params.limit {
            qp.push(("limit".to_string(), limit.to_string()));
        }
        if let Some(order) = params.order {
            qp.push(("order".to_string(), order.as_str().to_string()));
        }
        if let Some(after) = params.after {
            qp.push(("after".to_string(), after));
        }
        if let Some(include) = params.include {
            for item in include {
                qp.push(("include".to_string(), item));
            }
        }

        let url = if qp.is_empty() {
            url
        } else {
            self.with_query_params(&url, qp)?
        };

        let res = execute_get_request(&config, &url, None).await?;
        let page: crate::providers::openai::ext::responses::OpenAiResponsesInputItemsPage =
            serde_json::from_value(res.json).map_err(|e| {
                LlmError::ParseError(format!(
                    "Invalid OpenAI responses input_items response: {e}"
                ))
            })?;
        Ok(page)
    }

    pub(crate) async fn responses_compact(
        &self,
        request: crate::providers::openai::ext::responses::OpenAiResponsesCompactRequest,
    ) -> Result<crate::providers::openai::ext::responses::OpenAiResponsesCompaction, LlmError> {
        let config = self.responses_admin_config();
        let url = self.responses_url("responses/compact");
        let body = serde_json::to_value(request).map_err(|e| {
            LlmError::InvalidParameter(format!("Invalid OpenAI responses.compact request: {e}"))
        })?;

        let res = execute_json_request(&config, &url, HttpBody::Json(body), None, false).await?;
        let obj: crate::providers::openai::ext::responses::OpenAiResponsesCompaction =
            serde_json::from_value(res.json).map_err(|e| {
                LlmError::ParseError(format!("Invalid OpenAI responses.compact response: {e}"))
            })?;
        Ok(obj)
    }
}
