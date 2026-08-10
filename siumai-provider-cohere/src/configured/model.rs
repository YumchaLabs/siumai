use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;
use http::header::{ACCEPT, HeaderName, HeaderValue};
use http::{Method, StatusCode};
use siumai_core::{
    CallOptions, EmbeddingLimits, EmbeddingModel, EmbeddingRequest, EmbeddingResponse, Error,
    ErrorContext, ErrorKind, Model, ModelAdvisory, ModelDescriptor, ModelFamily, ModelId,
    ModelOperation, ModelPolicy, ModelPolicyContext, ProviderOptionError, PublicDiagnosticText,
    RerankLimits, RerankModel, RerankRequest, RerankResponse, RerankResult, ResponseMetadata,
    SensitiveResponse, SupportState, Usage, UsageValue, Warning, WarningKind,
};
use siumai_transport::{
    RequestBody, RequestBuildError, RequestHeaders, RequestPlan, RequestTarget, ResponseHeaders,
    TransportResponse,
};

use crate::provider_options::CohereEmbeddingInputType;

use super::options::{embedding_options, rerank_options};
use super::provider::CohereRuntime;
use super::wire::{
    EmbeddingWireRequest, EmbeddingWireResponse, RerankWireRequest, RerankWireResponse,
};

const EMBEDDING_TARGET: &str = "embed";
const RERANK_TARGET: &str = "rerank";
const MAX_EMBEDDING_INPUTS: usize = 96;
const MAX_RERANK_CANDIDATES: usize = 1000;
const ERROR_CAPTURE_BYTES: usize = 64 * 1024;
const VALID_OUTPUT_DIMENSIONS: &[u32] = &[256, 512, 1024, 1536];

/// Lightweight Cohere v2 embedding model handle.
#[derive(Clone)]
pub struct CohereEmbeddingModel {
    pub(crate) runtime: Arc<CohereRuntime>,
    descriptor: ModelDescriptor,
}

impl CohereEmbeddingModel {
    pub(crate) fn new(runtime: Arc<CohereRuntime>, model: ModelId) -> Self {
        let descriptor = ModelDescriptor::from_scope(
            runtime.scope.clone(),
            model,
            ModelFamily::Embedding,
            runtime.instance_id.clone(),
        );
        Self {
            runtime,
            descriptor,
        }
    }

    fn contextualize(&self, error: Error) -> Error {
        error.with_context(ErrorContext {
            operation: Some(ModelOperation::Embed),
            provider: Some(self.provider_id().clone()),
            route: None,
            model: Some(self.model_id().clone()),
        })
    }
}

impl std::fmt::Debug for CohereEmbeddingModel {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CohereEmbeddingModel")
            .field("descriptor", &self.descriptor)
            .finish()
    }
}

impl Model for CohereEmbeddingModel {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }
}

#[async_trait]
impl EmbeddingModel for CohereEmbeddingModel {
    fn limits(&self) -> EmbeddingLimits {
        EmbeddingLimits {
            max_inputs: Some(MAX_EMBEDDING_INPUTS),
            max_input_tokens: None,
        }
    }

    async fn embed(
        &self,
        request: EmbeddingRequest,
        options: CallOptions,
    ) -> Result<EmbeddingResponse, Error> {
        self.limits()
            .validate(&request)
            .map_err(|error| self.contextualize(error))?;
        let warnings = policy_warnings(&self.runtime, self.model_id(), ModelOperation::Embed)
            .map_err(|error| self.contextualize(error))?;
        let provider_options = embedding_options(&options, self.descriptor.scope())
            .map_err(option_error)
            .map_err(|error| self.contextualize(error))?;
        let dimensions =
            resolve_dimensions(self.model_id(), &request, provider_options.output_dimension)
                .map_err(|error| self.contextualize(error))?;
        let wire = EmbeddingWireRequest {
            model: self.model_id().as_str(),
            embedding_types: ["float"],
            texts: request.inputs(),
            input_type: provider_options
                .input_type
                .unwrap_or(CohereEmbeddingInputType::SearchQuery),
            truncate: provider_options.truncate,
            output_dimension: dimensions,
        };
        let response = self
            .runtime
            .transport
            .execute(
                json_plan(EMBEDDING_TARGET, &wire, self.runtime.replay_safety.clone())
                    .map_err(|error| self.contextualize(error))?,
                options,
            )
            .await
            .map_err(|error| self.contextualize(error))?;
        if !response.status().is_success() {
            return Err(self.contextualize(provider_status_error(response)));
        }
        let request_id = response_request_id(response.headers());
        let decoded =
            decode_embedding(response.body()).map_err(|error| self.contextualize(error))?;
        let result = EmbeddingResponse {
            embeddings: decoded.embeddings.float,
            metadata: ResponseMetadata {
                response_id: decoded.id,
                request_id,
                model: Some(self.model_id().clone()),
            },
            usage: embedding_usage(&decoded.meta),
            warnings,
            provider: provider_metadata(&decoded.meta),
        };
        validate_embedding_response(&result, &request, dimensions)
            .map_err(|error| self.contextualize(error))?;
        Ok(result)
    }
}

/// Lightweight Cohere v2 rerank model handle.
#[derive(Clone)]
pub struct CohereRerankModel {
    pub(crate) runtime: Arc<CohereRuntime>,
    descriptor: ModelDescriptor,
}

impl CohereRerankModel {
    pub(crate) fn new(runtime: Arc<CohereRuntime>, model: ModelId) -> Self {
        let descriptor = ModelDescriptor::from_scope(
            runtime.scope.clone(),
            model,
            ModelFamily::Rerank,
            runtime.instance_id.clone(),
        );
        Self {
            runtime,
            descriptor,
        }
    }

    fn contextualize(&self, error: Error) -> Error {
        error.with_context(ErrorContext {
            operation: Some(ModelOperation::Rerank),
            provider: Some(self.provider_id().clone()),
            route: None,
            model: Some(self.model_id().clone()),
        })
    }
}

impl std::fmt::Debug for CohereRerankModel {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CohereRerankModel")
            .field("descriptor", &self.descriptor)
            .finish()
    }
}

impl Model for CohereRerankModel {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }
}

#[async_trait]
impl RerankModel for CohereRerankModel {
    fn limits(&self) -> RerankLimits {
        RerankLimits {
            max_candidates: Some(MAX_RERANK_CANDIDATES),
        }
    }

    async fn rerank(
        &self,
        request: RerankRequest,
        options: CallOptions,
    ) -> Result<RerankResponse, Error> {
        self.limits()
            .validate(&request)
            .map_err(|error| self.contextualize(error))?;
        let warnings = policy_warnings(&self.runtime, self.model_id(), ModelOperation::Rerank)
            .map_err(|error| self.contextualize(error))?;
        let provider_options = rerank_options(&options, self.descriptor.scope())
            .map_err(option_error)
            .map_err(|error| self.contextualize(error))?;
        let wire = RerankWireRequest {
            model: self.model_id().as_str(),
            query: request.query(),
            documents: request
                .candidates()
                .iter()
                .map(|candidate| candidate.text())
                .collect(),
            top_n: request.top_n(),
            max_tokens_per_doc: provider_options.max_tokens_per_doc,
            priority: provider_options.priority,
        };
        let response = self
            .runtime
            .transport
            .execute(
                json_plan(RERANK_TARGET, &wire, self.runtime.replay_safety.clone())
                    .map_err(|error| self.contextualize(error))?,
                options,
            )
            .await
            .map_err(|error| self.contextualize(error))?;
        if !response.status().is_success() {
            return Err(self.contextualize(provider_status_error(response)));
        }
        let request_id = response_request_id(response.headers());
        let decoded = decode_rerank(response.body()).map_err(|error| self.contextualize(error))?;
        let mut results = Vec::with_capacity(decoded.results.len());
        for result in decoded.results {
            let candidate_id = request
                .candidates()
                .get(result.index)
                .and_then(|candidate| candidate.id())
                .map(str::to_owned);
            results.push(
                RerankResult::new(result.index, result.relevance_score, candidate_id)
                    .map_err(|error| self.contextualize(error))?,
            );
        }
        let result = RerankResponse {
            results,
            metadata: ResponseMetadata {
                response_id: decoded.id,
                request_id,
                model: Some(self.model_id().clone()),
            },
            usage: rerank_usage(&decoded.meta),
            warnings,
            provider: provider_metadata(&decoded.meta),
        };
        result
            .validate(&request)
            .map_err(|error| self.contextualize(error))?;
        Ok(result)
    }
}

fn json_plan(
    target: &str,
    body: &impl serde::Serialize,
    replay_safety: siumai_transport::ReplaySafety,
) -> Result<RequestPlan, Error> {
    let headers = RequestHeaders::new()
        .try_insert(ACCEPT, HeaderValue::from_static("application/json"))
        .map_err(request_build_error)?;
    RequestPlan::new(
        Method::POST,
        RequestTarget::new(target).map_err(request_build_error)?,
    )
    .with_headers(headers)
    .with_body(RequestBody::json(body).map_err(request_build_error)?)
    .with_replay_safety(replay_safety)
    .map_err(request_build_error)
}

fn resolve_dimensions(
    model: &ModelId,
    request: &EmbeddingRequest,
    option_dimensions: Option<u32>,
) -> Result<Option<u32>, Error> {
    let request_dimensions = request.dimensions().map(|value| value.get());
    if let (Some(requested), Some(option)) = (request_dimensions, option_dimensions)
        && requested != option
    {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "Cohere output dimension conflicts with the canonical embedding request",
        ));
    }
    let dimensions = request_dimensions.or(option_dimensions);
    if dimensions.is_some() && !crate::models::supports_output_dimension(model.as_str()) {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "Cohere output dimensions are supported only by Embed v4 models",
        ));
    }
    if let Some(dimensions) = dimensions
        && !VALID_OUTPUT_DIMENSIONS.contains(&dimensions)
    {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "Cohere output dimension must be 256, 512, 1024, or 1536",
        ));
    }
    Ok(dimensions)
}

fn validate_embedding_response(
    response: &EmbeddingResponse,
    request: &EmbeddingRequest,
    expected_dimensions: Option<u32>,
) -> Result<(), Error> {
    response.validate(request)?;
    if let Some(expected_dimensions) = expected_dimensions
        && response
            .embeddings
            .iter()
            .any(|embedding| embedding.len() != expected_dimensions as usize)
    {
        return Err(Error::protocol_violation(
            "Cohere embedding response does not match the requested output dimension",
        ));
    }
    Ok(())
}

fn policy_warnings(
    runtime: &CohereRuntime,
    model: &ModelId,
    operation: ModelOperation,
) -> Result<Vec<Warning>, Error> {
    let decision = runtime.policy.evaluate(&ModelPolicyContext::new(
        runtime.scope.clone(),
        model.clone(),
        operation,
    ));
    if matches!(decision.state(), SupportState::Unsupported { .. }) {
        return Err(Error::new(
            ErrorKind::Unsupported,
            "Cohere model policy rejected the requested operation",
        ));
    }
    Ok(decision
        .advisories()
        .iter()
        .map(|advisory| match advisory {
            ModelAdvisory::UnknownModel => Warning::new(
                WarningKind::UnknownModel,
                "model is absent from the verified Cohere advisory catalog",
            ),
            ModelAdvisory::Deprecated { .. } => {
                Warning::new(WarningKind::DeprecatedModel, "Cohere model is deprecated")
            }
            ModelAdvisory::Retired { .. } => {
                Warning::new(WarningKind::RetiredModel, "Cohere model is retired")
            }
            ModelAdvisory::RollingAlias => Warning::new(
                WarningKind::RollingModelAlias,
                "Cohere model ID is a rolling alias",
            ),
            _ => Warning::provider("model_advisory", "Cohere model policy returned an advisory"),
        })
        .collect())
}

fn decode_embedding(body: &[u8]) -> Result<EmbeddingWireResponse, Error> {
    serde_json::from_slice(body).map_err(|source| {
        Error::new(
            ErrorKind::Protocol,
            "Cohere returned an invalid embedding response",
        )
        .with_source(source)
    })
}

fn decode_rerank(body: &[u8]) -> Result<RerankWireResponse, Error> {
    serde_json::from_slice(body).map_err(|source| {
        Error::new(
            ErrorKind::Protocol,
            "Cohere returned an invalid rerank response",
        )
        .with_source(source)
    })
}

fn embedding_usage(meta: &serde_json::Value) -> Usage {
    Usage::default().with_input_tokens(usage_value(
        meta.get("billed_units")
            .and_then(|units| units.get("input_tokens"))
            .and_then(serde_json::Value::as_u64),
    ))
}

fn rerank_usage(meta: &serde_json::Value) -> Usage {
    let mut usage = Usage::default();
    if let Some(search_units) = meta
        .get("billed_units")
        .and_then(|units| units.get("search_units"))
        .and_then(serde_json::Value::as_u64)
    {
        usage
            .provider
            .insert("search_units".to_string(), search_units.into());
    }
    usage
}

fn usage_value(value: Option<u64>) -> UsageValue {
    value.map_or(UsageValue::Unknown, UsageValue::Known)
}

fn provider_metadata(meta: &serde_json::Value) -> BTreeMap<String, serde_json::Value> {
    BTreeMap::from([("cohere".to_string(), meta.clone())])
}

fn response_request_id(headers: &ResponseHeaders) -> Option<String> {
    headers
        .get(&HeaderName::from_static("x-request-id"))
        .and_then(|value| value.to_str().ok())
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
}

fn provider_status_error(response: TransportResponse) -> Error {
    let (status, headers, body) = response.into_parts();
    let kind = match status {
        StatusCode::UNAUTHORIZED => ErrorKind::Authentication,
        StatusCode::FORBIDDEN => ErrorKind::Authorization,
        StatusCode::TOO_MANY_REQUESTS => ErrorKind::RateLimited,
        _ => ErrorKind::Provider,
    };
    let raw_headers = headers
        .expose()
        .iter()
        .filter_map(|(name, value)| {
            value
                .to_str()
                .ok()
                .map(|value| (name.to_string(), value.to_string()))
        })
        .collect::<BTreeMap<_, _>>();
    let mut diagnostics = headers
        .diagnostics()
        .with_status(status.as_u16())
        .with_body_truncated(body.len() > ERROR_CAPTURE_BYTES);
    if let Some(request_id) = response_request_id(&headers)
        && let Ok(request_id) = PublicDiagnosticText::new(request_id)
    {
        diagnostics = diagnostics.with_request_id(request_id);
    }
    Error::new(kind, "Cohere rejected the provider request")
        .with_diagnostics(diagnostics)
        .with_sensitive_response(SensitiveResponse::with_limit(
            raw_headers,
            body.to_vec(),
            ERROR_CAPTURE_BYTES,
        ))
}

fn option_error(source: ProviderOptionError) -> Error {
    Error::new(
        ErrorKind::InvalidInput,
        "provider options are invalid for the selected Cohere model family",
    )
    .with_source(source)
}

fn request_build_error(source: RequestBuildError) -> Error {
    Error::new(
        ErrorKind::InvalidInput,
        "Cohere request violates the transport contract",
    )
    .with_source(source)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_dimensions_are_limited_to_embed_v4_models() {
        let request = EmbeddingRequest::single("hello").expect("embedding request");
        let v4 = ModelId::new(crate::models::embedding::EMBED_V4).expect("v4 model ID");
        let v3 = ModelId::new(crate::models::embedding::EMBED_ENGLISH_V3).expect("v3 model ID");

        assert_eq!(
            resolve_dimensions(&v4, &request, Some(512)).unwrap(),
            Some(512)
        );
        let error = resolve_dimensions(&v3, &request, Some(512)).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::InvalidInput);
    }
}
