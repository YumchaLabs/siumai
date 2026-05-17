use super::{DEPRECATED_OPENAI_COMPATIBLE_KEY_WARNING, OpenAiCompatibleClient};
use crate::core::{ProviderContext, ProviderSpec};
use crate::error::LlmError;
use crate::execution::executors::common::{HttpBody, HttpExecutionConfig};
use crate::standards::openai::completion_metadata::{
    completion_response_metadata, extract_completion_provider_metadata,
};
use crate::standards::openai::completion_request::{self, CompletionBodyOptions};
use crate::standards::openai::utils::{
    parse_provider_openai_finish_reason, parse_provider_openai_usage_value,
};
use crate::streaming::ChatStream;
use crate::traits::CompletionCapability;
use crate::types::{CompletionRequest, CompletionResponse, Warning};
use async_trait::async_trait;
use std::sync::Arc;

fn completion_provider_options_key(provider_id: &str) -> String {
    siumai_protocol_openai::standards::openai::compat::metadata::provider_options_key(provider_id)
}

mod streaming;

use streaming::CompletionSseConverter;

impl OpenAiCompatibleClient {
    fn prepare_completion_request(
        &self,
        mut request: CompletionRequest,
    ) -> Result<CompletionRequest, LlmError> {
        self.ensure_completion_surface(false)?;
        request.common_params = crate::utils::chat_request::merge_common_params(
            &self.config.common_params,
            request.common_params,
        );
        if request.http_config.is_none() {
            request.http_config = Some(self.config.http_config.clone());
        }
        if request.common_params.model.trim().is_empty() {
            return Err(LlmError::InvalidParameter(
                "OpenAI-compatible completion request requires a model".to_string(),
            ));
        }

        Ok(request)
    }

    fn completion_execution_config(
        &self,
        spec: Arc<dyn ProviderSpec>,
        ctx: ProviderContext,
    ) -> HttpExecutionConfig {
        HttpExecutionConfig {
            provider_id: self.config.provider_id.clone(),
            http_client: self.http_client.clone(),
            transport: self.config.http_transport.clone(),
            provider_spec: spec,
            provider_context: ctx,
            interceptors: self.http_interceptors.clone(),
            retry_options: self.retry_options.clone(),
        }
    }

    fn completion_url(&self) -> String {
        let base_url = self.config.adapter.url_for(
            &self.config.base_url,
            crate::providers::openai_compatible::RequestType::Completion,
        );
        crate::utils::url::with_query_params(&base_url, &self.config.query_params)
    }

    fn completion_provider_options(
        &self,
        request: &CompletionRequest,
    ) -> serde_json::Map<String, serde_json::Value> {
        let mut merged = serde_json::Map::new();

        for options in [Some("openai-compatible"), Some("openaiCompatible")]
            .into_iter()
            .flatten()
            .filter_map(|key| request.provider_options_map.get_object(key))
            .chain(
                siumai_protocol_openai::standards::openai::compat::metadata::provider_options_keys(
                    &self.config.provider_id,
                )
                .into_iter()
                .filter_map(|key| request.provider_options_map.get_object(&key)),
            )
        {
            for (key, value) in options {
                merged.insert(key.clone(), value.clone());
            }
        }

        if let Some(logit_bias) = merged.remove("logitBias") {
            merged.entry("logit_bias".to_string()).or_insert(logit_bias);
        }

        merged
    }

    fn build_completion_body(
        &self,
        request: &CompletionRequest,
        stream: bool,
    ) -> Result<(serde_json::Value, Vec<Warning>), LlmError> {
        let (mut body, warnings) = completion_request::build_completion_body(
            request,
            CompletionBodyOptions::new(stream)
                .with_include_usage(stream && self.config.include_usage == Some(true))
                .with_deprecated_openai_compatible_key_warning(Some(
                    DEPRECATED_OPENAI_COMPATIBLE_KEY_WARNING,
                ))
                .with_provider_options(self.completion_provider_options(request)),
        )?;
        self.config.adapter.transform_request_params(
            &mut body,
            &request.common_params.model,
            crate::providers::openai_compatible::RequestType::Completion,
        )?;
        if let Some(transformer) = self.request_settings().request_body_transformer {
            transformer.transform_request_body(
                &mut body,
                &request.common_params.model,
                crate::providers::openai_compatible::RequestType::Completion,
            )?;
        }

        Ok((body, warnings))
    }

    fn build_completion_response(
        &self,
        raw: serde_json::Value,
        headers: &reqwest::header::HeaderMap,
        warnings: Vec<Warning>,
    ) -> CompletionResponse {
        let provider_metadata_key = completion_provider_options_key(&self.config.provider_id);
        let text = raw
            .get("choices")
            .and_then(|value| value.as_array())
            .and_then(|choices| choices.first())
            .and_then(|choice| choice.get("text"))
            .and_then(|value| value.as_str())
            .map(ToString::to_string)
            .unwrap_or_default();
        let raw_finish_reason = raw
            .get("choices")
            .and_then(|value| value.as_array())
            .and_then(|choices| choices.first())
            .and_then(|choice| choice.get("finish_reason"))
            .and_then(|value| value.as_str())
            .map(ToString::to_string);
        let finish_reason = raw_finish_reason.as_deref().and_then(|value| {
            parse_provider_openai_finish_reason(self.config.provider_id.as_str(), Some(value))
        });

        CompletionResponse {
            text,
            finish_reason,
            raw_finish_reason,
            usage: raw.get("usage").and_then(|usage| {
                parse_provider_openai_usage_value(self.config.provider_id.as_str(), usage)
            }),
            response_metadata: Some(completion_response_metadata(
                self.config.provider_id.clone(),
                &raw,
                headers,
                true,
            )),
            warnings: (!warnings.is_empty()).then_some(warnings),
            provider_metadata: extract_completion_provider_metadata(&provider_metadata_key, &raw),
        }
    }

    async fn completion_request_via_spec(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, LlmError> {
        let request = self.prepare_completion_request(request)?;
        let (body, warnings) = self.build_completion_body(&request, false)?;
        let ctx = self.build_context().await?;
        let spec = Arc::new(self.compat_spec());
        let config = self.completion_execution_config(spec.clone(), ctx);
        let url = self.completion_url();

        let result = crate::execution::executors::http_request::execute_json_request(
            &config,
            &url,
            HttpBody::Json(body),
            request.http_config.as_ref(),
            false,
        )
        .await?;

        Ok(self.build_completion_response(result.json, &result.headers, warnings))
    }

    async fn completion_stream_request_via_spec(
        &self,
        request: CompletionRequest,
    ) -> Result<ChatStream, LlmError> {
        self.ensure_completion_surface(true)?;
        let request = self.prepare_completion_request(request)?;
        let (body, warnings) = self.build_completion_body(&request, true)?;
        let disable_compression = request
            .http_config
            .as_ref()
            .map(|config| config.stream_disable_compression)
            .unwrap_or(false);
        let ctx = self.build_context().await?;
        let spec = Arc::new(self.compat_spec());
        let headers_base = spec.build_headers(&ctx)?;
        let url = self.completion_url();
        let request_id = crate::execution::http::interceptor::generate_request_id();
        let converter = crate::streaming::InterceptingConverter {
            interceptors: self.http_interceptors.clone(),
            ctx: crate::execution::http::interceptor::HttpRequestContext {
                request_id: request_id.clone(),
                provider_id: self.config.provider_id.clone(),
                url: url.clone(),
                stream: true,
            },
            convert: CompletionSseConverter::new(
                self.config.provider_id.clone(),
                completion_provider_options_key(&self.config.provider_id),
                warnings,
                request.stream_options.include_raw_chunks,
            ),
        };

        crate::execution::executors::stream_sse::execute_sse_stream_request_with_headers(
            &self.http_client,
            &self.config.provider_id,
            Some(spec.as_ref()),
            &url,
            request_id,
            headers_base,
            body,
            &self.http_interceptors,
            self.retry_options.clone(),
            request.http_config,
            converter,
            disable_compression,
            self.config.http_transport.clone(),
        )
        .await
    }
}

#[async_trait]
impl CompletionCapability for OpenAiCompatibleClient {
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, LlmError> {
        self.completion_request_via_spec(request).await
    }

    async fn complete_stream(&self, request: CompletionRequest) -> Result<ChatStream, LlmError> {
        self.completion_stream_request_via_spec(request).await
    }
}

#[cfg(test)]
mod tests;
