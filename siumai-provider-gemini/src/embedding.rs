use std::sync::Arc;

use async_trait::async_trait;
use http::Method;
use http::header::{ACCEPT, HeaderValue};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use siumai_core::{
    CallOptions, EmbeddingLimits, EmbeddingModel, EmbeddingRequest, EmbeddingResponse, Error,
    ErrorContext, ErrorKind, Model, ModelDescriptor, ModelFamily, ModelId, ModelOperation,
    ProviderOptionError, ProviderOptionSelection, ProviderOptions, ProviderScope,
    TypedProviderOptions,
};
use siumai_protocol_gemini::embedding::{
    EmbedContentConfig, EmbedContentTaskType, EmbeddingRequestMode, decode_embedding_response,
    encode_embedding_request,
};
use siumai_transport::{ReplaySafety, RequestBody, RequestHeaders, RequestPlan, RequestTarget};

use crate::http::{response_error, response_request_id};
use crate::provider::ProviderRuntime;

/// Provider API mode used by stable v1 synchronous Gemini embeddings.
pub const GEMINI_EMBEDDING_API_MODE_ID: &str = "embed-content-v1";

/// Current Gemini embedding model hint.
pub const GEMINI_EMBEDDING_2: &str = "gemini-embedding-2";
/// Stable text-only Gemini embedding compatibility hint.
pub const GEMINI_EMBEDDING_001: &str = "gemini-embedding-001";
const KNOWN_MAX_INPUT_TOKENS: u64 = 8_192;
const MAX_TITLE_BYTES: usize = 16 * 1024;

/// Task types supported by stable Gemini `EmbedContentConfig`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
#[non_exhaustive]
pub enum GeminiEmbeddingTaskType {
    /// Embed a query for asymmetric retrieval.
    RetrievalQuery,
    /// Embed a document for asymmetric retrieval.
    RetrievalDocument,
    /// Embed text for symmetric semantic comparison.
    SemanticSimilarity,
    /// Embed text for classification.
    Classification,
    /// Embed text for clustering.
    Clustering,
    /// Embed text for question answering.
    QuestionAnswering,
    /// Embed text for fact verification.
    FactVerification,
    /// Embed a natural-language query for code retrieval.
    CodeRetrievalQuery,
}

/// Provider-owned controls for stable v1 Gemini text embeddings.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct GeminiEmbeddingOptions {
    /// Optional task type. Current `gemini-embedding-2` rejects this field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task_type: Option<GeminiEmbeddingTaskType>,
    /// Optional retrieval-document title.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// Whether the provider may silently truncate overlong input. Defaults to `false`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auto_truncate: Option<bool>,
}

impl GeminiEmbeddingOptions {
    pub const fn new() -> Self {
        Self {
            task_type: None,
            title: None,
            auto_truncate: None,
        }
    }

    pub const fn with_task_type(mut self, task_type: GeminiEmbeddingTaskType) -> Self {
        self.task_type = Some(task_type);
        self
    }

    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    pub const fn with_auto_truncate(mut self, auto_truncate: bool) -> Self {
        self.auto_truncate = Some(auto_truncate);
        self
    }
}

impl TypedProviderOptions for GeminiEmbeddingOptions {
    const NAMESPACE: &'static str = "google";
    const MODEL_FAMILY: ModelFamily = ModelFamily::Embedding;
    const API_MODE: Option<&'static str> = Some(GEMINI_EMBEDDING_API_MODE_ID);

    fn validate(&self) -> Result<(), ProviderOptionError> {
        if let Some(title) = self.title.as_deref()
            && (title.trim().is_empty() || title.len() > MAX_TITLE_BYTES)
        {
            return Err(ProviderOptionError::Rejected {
                path: "title".to_string(),
                reason: "must be non-empty and no more than 16384 bytes".to_string(),
            });
        }
        Ok(())
    }
}

/// Lightweight stable-v1 Gemini text embedding handle.
#[derive(Clone)]
pub struct GeminiEmbeddingModel {
    runtime: Arc<ProviderRuntime>,
    descriptor: ModelDescriptor,
    defaults: GeminiEmbeddingOptions,
}

impl GeminiEmbeddingModel {
    pub(crate) fn new(
        runtime: Arc<ProviderRuntime>,
        scope: Arc<ProviderScope>,
        model: ModelId,
        defaults: GeminiEmbeddingOptions,
    ) -> Self {
        let descriptor = ModelDescriptor::from_scope(
            scope,
            model,
            ModelFamily::Embedding,
            runtime.instance_id.clone(),
        );
        Self {
            runtime,
            descriptor,
            defaults,
        }
    }

    fn options(&self, call: &CallOptions) -> Result<GeminiEmbeddingOptions, Error> {
        let selection = call.provider_options_for(self).map_err(option_error)?;
        merge_options(&self.defaults, &selection).map_err(option_error)
    }

    fn plan(
        &self,
        request: &EmbeddingRequest,
        options: &GeminiEmbeddingOptions,
    ) -> Result<(RequestPlan, EmbeddingRequestMode), Error> {
        let encoded =
            encode_embedding_request(request, self.model_id(), &protocol_config(options))?;
        let (target, body, mode) = encoded.into_parts();
        let headers = RequestHeaders::new()
            .try_insert(ACCEPT, HeaderValue::from_static("application/json"))
            .map_err(request_build_error)?;
        let plan = RequestPlan::new(
            Method::POST,
            RequestTarget::new(target).map_err(request_build_error)?,
        )
        .with_headers(headers)
        .with_body(RequestBody::json(&body).map_err(request_build_error)?)
        .with_replay_safety(ReplaySafety::SemanticallyIdempotent)
        .map_err(request_build_error)?;
        Ok((plan, mode))
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

impl std::fmt::Debug for GeminiEmbeddingModel {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("GeminiEmbeddingModel")
            .field("descriptor", &self.descriptor)
            .finish()
    }
}

impl Model for GeminiEmbeddingModel {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }
}

#[async_trait]
impl EmbeddingModel for GeminiEmbeddingModel {
    fn limits(&self) -> EmbeddingLimits {
        EmbeddingLimits {
            max_inputs: None,
            max_input_tokens: is_known_model(self.model_id()).then_some(KNOWN_MAX_INPUT_TOKENS),
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
        let provider_options = self
            .options(&options)
            .map_err(|error| self.contextualize(error))?;
        let (plan, mode) = self
            .plan(&request, &provider_options)
            .map_err(|error| self.contextualize(error))?;
        let response = self
            .runtime
            .transport
            .execute(plan, options)
            .await
            .map_err(|error| self.contextualize(error))?;
        if !response.status().is_success() {
            return Err(self.contextualize(response_error(
                response,
                "Gemini rejected the stable v1 embedding request",
            )));
        }

        let (_, headers, body) = response.into_parts();
        let mut decoded = decode_embedding_response(&body, mode, &request, self.model_id())
            .map_err(|error| self.contextualize(error))?;
        decoded.metadata.request_id = response_request_id(&headers);
        Ok(decoded)
    }
}

fn protocol_config(options: &GeminiEmbeddingOptions) -> EmbedContentConfig {
    let mut config =
        EmbedContentConfig::new().with_auto_truncate(options.auto_truncate.unwrap_or(false));
    if let Some(task_type) = options.task_type {
        config = config.with_task_type(protocol_task_type(task_type));
    }
    if let Some(title) = options.title.as_deref() {
        config = config.with_title(title);
    }
    config
}

fn protocol_task_type(task_type: GeminiEmbeddingTaskType) -> EmbedContentTaskType {
    match task_type {
        GeminiEmbeddingTaskType::RetrievalQuery => EmbedContentTaskType::RetrievalQuery,
        GeminiEmbeddingTaskType::RetrievalDocument => EmbedContentTaskType::RetrievalDocument,
        GeminiEmbeddingTaskType::SemanticSimilarity => EmbedContentTaskType::SemanticSimilarity,
        GeminiEmbeddingTaskType::Classification => EmbedContentTaskType::Classification,
        GeminiEmbeddingTaskType::Clustering => EmbedContentTaskType::Clustering,
        GeminiEmbeddingTaskType::QuestionAnswering => EmbedContentTaskType::QuestionAnswering,
        GeminiEmbeddingTaskType::FactVerification => EmbedContentTaskType::FactVerification,
        GeminiEmbeddingTaskType::CodeRetrievalQuery => EmbedContentTaskType::CodeRetrievalQuery,
    }
}

fn is_known_model(model: &ModelId) -> bool {
    matches!(model.as_str(), GEMINI_EMBEDDING_2 | GEMINI_EMBEDDING_001)
}

fn merge_options(
    defaults: &GeminiEmbeddingOptions,
    selection: &ProviderOptionSelection<'_>,
) -> Result<GeminiEmbeddingOptions, ProviderOptionError> {
    let mut merged = defaults.clone();
    for options in selection.typed() {
        let value = decode_options(options)?;
        value.validate()?;
        if value.task_type.is_some() {
            merged.task_type = value.task_type;
        }
        if value.title.is_some() {
            merged.title = value.title;
        }
        if value.auto_truncate.is_some() {
            merged.auto_truncate = value.auto_truncate;
        }
    }
    if selection.raw_override().is_some() {
        return Err(ProviderOptionError::Rejected {
            path: "$".to_string(),
            reason: "Gemini stable v1 embedding only accepts typed provider options".to_string(),
        });
    }
    merged.validate()?;
    Ok(merged)
}

fn decode_options(
    options: &ProviderOptions,
) -> Result<GeminiEmbeddingOptions, ProviderOptionError> {
    serde_json::from_value(Value::Object(options.value().clone())).map_err(|_| {
        ProviderOptionError::Rejected {
            path: "google".to_string(),
            reason: "options do not match the Gemini stable v1 embedding schema".to_string(),
        }
    })
}

fn option_error(source: ProviderOptionError) -> Error {
    Error::new(
        ErrorKind::InvalidInput,
        "provider options are invalid for Gemini stable v1 embedding",
    )
    .with_source(source)
}

fn request_build_error(source: siumai_transport::RequestBuildError) -> Error {
    Error::new(
        ErrorKind::InvalidInput,
        "Gemini stable v1 embedding request violates the transport contract",
    )
    .with_source(source)
}
