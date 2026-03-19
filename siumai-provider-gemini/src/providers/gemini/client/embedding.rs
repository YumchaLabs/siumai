use super::GeminiClient;
use crate::error::LlmError;
use crate::traits::{EmbeddingCapability, EmbeddingExtensions};
use crate::types::{
    BatchEmbeddingRequest, BatchEmbeddingResponse, EmbeddingRequest, EmbeddingResponse, HttpConfig,
};
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;

fn effective_embedding_model(request: &EmbeddingRequest, default_model: &str) -> String {
    request
        .model
        .clone()
        .filter(|model| !model.trim().is_empty())
        .unwrap_or_else(|| default_model.to_string())
}

fn http_configs_match(left: Option<&HttpConfig>, right: Option<&HttpConfig>) -> bool {
    match (left, right) {
        (None, None) => true,
        (Some(left), Some(right)) => {
            left.timeout == right.timeout
                && left.connect_timeout == right.connect_timeout
                && left.headers == right.headers
                && left.proxy == right.proxy
                && left.user_agent == right.user_agent
                && left.stream_disable_compression == right.stream_disable_compression
        }
        _ => false,
    }
}

fn requests_can_be_coalesced(
    baseline: &EmbeddingRequest,
    candidate: &EmbeddingRequest,
    default_model: &str,
) -> bool {
    !baseline.input.is_empty()
        && !candidate.input.is_empty()
        && effective_embedding_model(baseline, default_model)
            == effective_embedding_model(candidate, default_model)
        && baseline.dimensions == candidate.dimensions
        && baseline.encoding_format == candidate.encoding_format
        && baseline.user == candidate.user
        && baseline.task_type == candidate.task_type
        && baseline.title == candidate.title
        && baseline.provider_options_map.0 == candidate.provider_options_map.0
        && http_configs_match(
            baseline.http_config.as_ref(),
            candidate.http_config.as_ref(),
        )
}

fn coalesce_batch_requests(
    requests: &[EmbeddingRequest],
    default_model: &str,
) -> Option<(EmbeddingRequest, Vec<usize>)> {
    let baseline = requests.first()?;
    if requests
        .iter()
        .skip(1)
        .any(|request| !requests_can_be_coalesced(baseline, request, default_model))
    {
        return None;
    }

    let lengths = requests.iter().map(|request| request.input.len()).collect();
    let mut merged = baseline.clone();
    merged.model = Some(effective_embedding_model(baseline, default_model));
    merged.input = requests
        .iter()
        .flat_map(|request| request.input.iter().cloned())
        .collect();
    Some((merged, lengths))
}

fn split_coalesced_response(
    response: EmbeddingResponse,
    lengths: &[usize],
) -> Result<BatchEmbeddingResponse, LlmError> {
    let total_inputs: usize = lengths.iter().sum();
    if total_inputs != response.embeddings.len() {
        return Err(LlmError::ParseError(format!(
            "Gemini batch embedding returned {} vectors for {} flattened inputs",
            response.embeddings.len(),
            total_inputs
        )));
    }

    let mut index = 0usize;
    let mut responses = Vec::with_capacity(lengths.len());
    for len in lengths {
        let next = index + len;
        responses.push(Ok(EmbeddingResponse {
            embeddings: response.embeddings[index..next].to_vec(),
            model: response.model.clone(),
            usage: None,
            metadata: response.metadata.clone(),
            response: None,
        }));
        index = next;
    }

    let mut metadata = HashMap::new();
    metadata.insert("coalesced".to_string(), serde_json::Value::Bool(true));
    if let Some(usage) = response.usage {
        metadata.insert(
            "aggregated_usage".to_string(),
            serde_json::json!({
                "prompt_tokens": usage.prompt_tokens,
                "total_tokens": usage.total_tokens,
            }),
        );
    }
    if let Some(http_response) = response.response {
        metadata.insert(
            "response".to_string(),
            serde_json::to_value(http_response).map_err(|error| {
                LlmError::ParseError(format!(
                    "Serialize Gemini batch response envelope failed: {error}"
                ))
            })?,
        );
    }

    Ok(BatchEmbeddingResponse {
        responses,
        metadata,
    })
}

#[async_trait]
impl EmbeddingCapability for GeminiClient {
    async fn embed(&self, texts: Vec<String>) -> Result<EmbeddingResponse, LlmError> {
        use crate::execution::executors::embedding::{EmbeddingExecutor, EmbeddingExecutorBuilder};
        let req = EmbeddingRequest::new(texts).with_model(self.config.model.clone());
        let ctx = super::super::context::build_context(&self.config).await;
        let spec = Arc::new(crate::providers::gemini::spec::GeminiSpecWithConfig::new(
            self.config.clone(),
        ));

        let exec = EmbeddingExecutorBuilder::new("gemini", self.http_client.clone())
            .with_spec(spec)
            .with_context(ctx)
            .with_interceptors(self.http_interceptors.clone());
        let exec = if let Some(transport) = self.config.http_transport.clone() {
            exec.with_transport(transport)
        } else {
            exec
        };

        let exec = if let Some(retry) = self.retry_options.clone() {
            exec.with_retry_options(retry).build_for_request(&req)
        } else {
            exec.build_for_request(&req)
        };

        EmbeddingExecutor::execute(&*exec, req).await
    }

    fn as_embedding_extensions(&self) -> Option<&dyn EmbeddingExtensions> {
        Some(self)
    }

    fn embedding_dimension(&self) -> usize {
        3072
    }
    fn max_tokens_per_embedding(&self) -> usize {
        2048
    }
    fn supported_embedding_models(&self) -> Vec<String> {
        vec!["gemini-embedding-001".to_string()]
    }
}

#[async_trait]
impl EmbeddingExtensions for GeminiClient {
    async fn embed_with_config(
        &self,
        request: EmbeddingRequest,
    ) -> Result<EmbeddingResponse, LlmError> {
        use crate::execution::executors::embedding::{EmbeddingExecutor, EmbeddingExecutorBuilder};
        let mut request = request;
        if request.input.len() > 2048 {
            return Err(LlmError::InvalidParameter(format!(
                "Too many values for a single embedding call. The Gemini model \"{}\" can only embed up to 2048 values per call, but {} values were provided.",
                self.config.model,
                request.input.len()
            )));
        }
        if request.model.is_none() {
            request.model = Some(self.config.model.clone());
        }
        let ctx = super::super::context::build_context(&self.config).await;
        let spec = Arc::new(crate::providers::gemini::spec::GeminiSpecWithConfig::new(
            self.config.clone(),
        ));

        let exec = EmbeddingExecutorBuilder::new("gemini", self.http_client.clone())
            .with_spec(spec)
            .with_context(ctx)
            .with_interceptors(self.http_interceptors.clone());
        let exec = if let Some(transport) = self.config.http_transport.clone() {
            exec.with_transport(transport)
        } else {
            exec
        };

        let exec = if let Some(retry) = self.retry_options.clone() {
            exec.with_retry_options(retry).build_for_request(&request)
        } else {
            exec.build_for_request(&request)
        };

        EmbeddingExecutor::execute(&*exec, request).await
    }

    async fn embed_batch(
        &self,
        requests: BatchEmbeddingRequest,
    ) -> Result<BatchEmbeddingResponse, LlmError> {
        if requests.requests.is_empty() {
            return Ok(BatchEmbeddingResponse {
                responses: Vec::new(),
                metadata: HashMap::new(),
            });
        }

        if let Some((merged_request, lengths)) =
            coalesce_batch_requests(&requests.requests, &self.config.model)
        {
            match self.embed_with_config(merged_request).await {
                Ok(response) => return split_coalesced_response(response, &lengths),
                Err(_) => {}
            }
        }

        let mut responses = Vec::new();
        for request in requests.requests {
            let result = self
                .embed_with_config(request)
                .await
                .map_err(|error| error.to_string());
            responses.push(result);
            if requests.batch_options.fail_fast && responses.last().is_some_and(|r| r.is_err()) {
                break;
            }
        }

        Ok(BatchEmbeddingResponse {
            responses,
            metadata: HashMap::new(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::ProviderSpec;

    #[test]
    fn spec_with_config_uses_request_model_for_embedding_body() {
        let base = crate::providers::gemini::GeminiConfig::default()
            .with_model("chat-model".to_string())
            .with_common_params(crate::types::CommonParams {
                model: "chat-model".to_string(),
                ..Default::default()
            });
        let spec = crate::providers::gemini::spec::GeminiSpecWithConfig::new(base);
        let ctx = crate::core::ProviderContext::new(
            "gemini",
            "https://example".to_string(),
            Some("KEY".to_string()),
            std::collections::HashMap::new(),
        );

        let req =
            EmbeddingRequest::new(vec!["hi".to_string()]).with_model("embed-model".to_string());
        let bundle = spec.choose_embedding_transformers(&req, &ctx);
        let body = bundle.request.transform_embedding(&req).unwrap();
        assert_eq!(body["model"], serde_json::json!("models/embed-model"));
    }
}
