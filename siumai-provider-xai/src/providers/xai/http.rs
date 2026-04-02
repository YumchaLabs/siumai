use super::XaiClient;
use crate::core::{ProviderContext, ProviderSpec};
use crate::error::LlmError;
use crate::execution::executors::common::HttpExecutionConfig;
use crate::traits::ProviderCapabilities;
use reqwest::header::HeaderMap;
use std::collections::HashMap;
use std::sync::Arc;

pub(super) const PROVIDER_ID: &str = "xai";

#[derive(Clone, Copy, Default)]
pub(super) struct XaiApiSpec;

impl ProviderSpec for XaiApiSpec {
    fn id(&self) -> &'static str {
        PROVIDER_ID
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new()
            .with_image_generation()
            .with_custom_feature("video", true)
    }

    fn build_headers(&self, ctx: &ProviderContext) -> Result<HeaderMap, LlmError> {
        siumai_protocol_openai::standards::openai::headers::build_openai_compatible_json_headers(
            ctx,
        )
    }

    fn classify_http_error(
        &self,
        status: u16,
        body_text: &str,
        _headers: &HeaderMap,
    ) -> Option<LlmError> {
        siumai_protocol_openai::standards::openai::errors::classify_openai_compatible_http_error(
            PROVIDER_ID,
            status,
            body_text,
        )
    }
}

pub(super) fn build_http_execution_config(client: &XaiClient) -> HttpExecutionConfig {
    HttpExecutionConfig {
        provider_id: PROVIDER_ID.to_string(),
        http_client: client.http_client(),
        transport: client.http_transport(),
        provider_spec: Arc::new(XaiApiSpec),
        provider_context: client.provider_context(),
        interceptors: client.http_interceptors(),
        retry_options: client.retry_options(),
    }
}

pub(super) fn headers_to_map(headers: &HeaderMap) -> HashMap<String, String> {
    headers
        .iter()
        .filter_map(|(key, value)| {
            Some((key.as_str().to_string(), value.to_str().ok()?.to_string()))
        })
        .collect()
}
